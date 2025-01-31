#!/bin/bash
NODE_RANK=$1  # 接受节点等级参数

###############################
### 环境变量与系统配置
###############################
export NUMEXPR_MAX_THREADS=128       # 数值计算最大线程数
export RAY_DEDUP_LOGS=0              # 禁用Ray日志去重


wandb_token="fafd69135210d3684f64d5676b5933814655295e"         
DATA_PATH="/home/t2vg-a100-G4-43/mem-kk-logic/kk_train_data/3ppl.jsonl"   # 训练数据路径

###############################
### 模型与训练参数
###############################
MODEL_NAME="Qwen/Qwen2.5-Math-7B"  # 基础模型名称
REWARD_MODEL="kk_server"       # 奖励模型标识符
SAVE_MODEL_NAME=$(echo "${MODEL_NAME}_${REWARD_MODEL}" | sed 's|/|_|g')  # 保存模型名称（替换斜杠）

# 采样与训练控制
N_SAMPLES=8                          # 每个提示的采样数
EPISODE=10000                        # 训练轮次
WARMUP=0.0                           # 学习率预热比例
TBS=512                              # 训练批次大小
RBS=128                              # rollout批次大小
KL=0.001                             # KL散度系数
LR=2e-6                              # 演员网络学习率
MAX_LENGTH=29000                     # 最大生成长度
PORT=1278                            # 奖励模型服务端口
TEMP=1.0                             # 采样温度

###############################
### 辅助配置
###############################
GROUP_METHOD="normal"                # 分组方法 (grpo/其他)
LOG_BASE="log"                       # 日志根目录

###############################
### 目录初始化
###############################
mkdir -p "results/${SAVE_MODEL_NAME}/server"  # 创建结果目录
mkdir -p "${LOG_BASE}/server/"                # 创建日志目录

###############################
### Ray集群管理
###############################
ray stop
sleep 2  # 等待Ray完全停止

# 启动Ray集群
ray start --head \
    --port=6379 \
    --dashboard-port=8265 \
    --num-gpus=4 \

# 等待Ray集群完全启动
sleep 5

###############################
### 奖励模型服务管理
###############################
# 终止现有奖励模型服务
pkill -f "${REWARD_MODEL}"

# 确保日志目录存在
mkdir -p "$(dirname "${LOG_BASE}/server/${SAVE_MODEL_NAME}-node${NODE_RANK}.log")"

# 启动奖励模型服务并等待它完全启动
nohup python -m "openrlhf.cli.${REWARD_MODEL}" \
  --data_path "${DATA_PATH}" \
  --reward_pretrain "${MODEL_NAME}" \
  --log_file "results/${SAVE_MODEL_NAME}/server/sampling.jsonl" \
  --port "${PORT}" \
  > "${LOG_BASE}/server/${SAVE_MODEL_NAME}-node${NODE_RANK}.log" 2>&1 &

# 等待服务器启动
echo "Waiting for reward model server to start..."
sleep 5  # 增加等待时间

# 使用get_reward端点检查服务器状态
check_server() {
  curl -s -X POST "http://localhost:${PORT}/get_reward" \
    -H "Content-Type: application/json" \
    -d '{"query": ["test"]}' > /dev/null 2>&1
}

until check_server; do
  echo "Waiting for server to be ready..."
  sleep 2
done
echo "Reward model server is ready"

# 输出日志路径
echo "reward model log: ${LOG_BASE}/server/${SAVE_MODEL_NAME}-node${NODE_RANK}.log"

###############################
### 主训练流程 (仅头节点执行)
###############################
if [ "${NODE_RANK}" = "0" ]; then
  # 检查Ray集群状态
  if ! ray status > /dev/null 2>&1; then
    echo "Error: Ray cluster is not running. Please check Ray initialization."
    exit 1
  fi

  # 确保奖励模型服务器正在运行
  if ! check_server; then
    echo "Error: Reward model server is not running"
    exit 1
  fi

  ray job submit --address="http://127.0.0.1:8265" -- \
    python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 4 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 4 \
      --vllm_num_engines 4 \
      --vllm_tensor_parallel_size 1 \
      --colocate_actor_ref \
      --pretrain "${MODEL_NAME}" \
      --remote_rm_url "http://localhost:${PORT}/get_reward" \
      --save_path "results/${SAVE_MODEL_NAME}" \
      --ckpt_path "results/${SAVE_MODEL_NAME}" \
      --micro_train_batch_size 1 \
      --train_batch_size "${TBS}" \
      --micro_rollout_batch_size 2 \
      --rollout_batch_size "${RBS}" \
      --advantage_estimator group_norm \
      --max_samples 100000 \
      --max_epochs 1 \
      --num_episodes "${EPISODE}" \
      --lr_warmup_ratio "${WARMUP}" \
      --n_samples_per_prompt "${N_SAMPLES}" \
      --prompt_max_len 1024 \
      --generate_max_len "${MAX_LENGTH}" \
      --bf16 \
      --actor_learning_rate "${LR}" \
      --init_kl_coef "${KL}" \
      --prompt_data "${DATA_PATH}" \
      --input_key quiz \
      --apply_chat_template \
      --packing_samples \
      --flash_attn \
      --gradient_checkpointing \
      --save_steps 10 \
      --use_wandb "${wandb_token}" \
      --wandb_run_name "${SAVE_MODEL_NAME}" \
      --wandb_project "qwen_grpo" \
      --vllm_sync_backend nccl \
      --max_ckpt_num 20 \
      --group_method "${GROUP_METHOD}" \
      --use_length_reward_in_efficiency \
      --temperature "${TEMP}" \
      --overlap_comm \
      # 可选参数:
      # --enable_ema \
      # --load_checkpoint
fi

# 在脚本退出时清理进程
cleanup() {
  pkill -f "${REWARD_MODEL}"
  ray stop
}
trap cleanup EXIT
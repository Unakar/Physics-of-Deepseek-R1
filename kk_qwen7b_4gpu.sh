#!/bin/bash
NODE_RANK=$1  # 接受节点等级参数

###############################
### 环境变量与系统配置
###############################
export NUMEXPR_MAX_THREADS=128       # 数值计算最大线程数
export RAY_DEDUP_LOGS=0              # 禁用Ray日志去重

###############################
### 关键凭证与路径配置
###############################
wandb_token="fafd69135210d3684f64d5676b5933814655295e"                    # WandB鉴权令牌
DATA_PATH="/home/t2vg-a100-G4-43/mem-kk-logic/kk_train_data/3ppl.jsonl"   # 训练数据路径

###############################
### 模型与训练参数
###############################
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 基础模型名称
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

ray stop

###############################
### 奖励模型服务管理
###############################
# 终止现有奖励模型服务
pkill -f "${REWARD_MODEL}"

# 确保日志目录存在
mkdir -p "$(dirname "${LOG_BASE}/server/${SAVE_MODEL_NAME}-node${NODE_RANK}.log")"

# 启动奖励模型服务
nohup python -m "openrlhf.cli.${REWARD_MODEL}" \
  --data_path "${DATA_PATH}" \
  --reward_pretrain "${MODEL_NAME}" \
  --log_file "results/${SAVE_MODEL_NAME}/server/sampling.jsonl" \
  --port "${PORT}" \
  > "${LOG_BASE}/server/${SAVE_MODEL_NAME}-node${NODE_RANK}.log" 2>&1 &

# 输出日志路径
echo "reward model log: ${LOG_BASE}/server/${SAVE_MODEL_NAME}-node${NODE_RANK}.log"

###############################
### 主训练流程 (仅头节点执行)
###############################
if [ "${NODE_RANK}" = "0" ]; then
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
      --zero_stage 2 \
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
      --overlap_comm
      # 可选参数:
      # --enable_ema \
      # --load_checkpoint
fi
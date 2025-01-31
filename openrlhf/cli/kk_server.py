import argparse
import re
import json
from typing import List, Dict, Tuple
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import datasets
from transformers import AutoTokenizer
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

DEFAULT_PORT = 5001
DEFAULT_HOST = "0.0.0.0"

class KKSolutionParser:
    @staticmethod
    def extract_components(response: str) -> Tuple[str, str]:
        """提取<think>和<answer>部分"""
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        
        thought = think_match.group(1).strip() if think_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        return thought, answer

    @staticmethod
    def parse_query(query: str) -> Tuple[str, str]:
        """解析用户查询"""
        question_part = query.split("### Question:")[-1].split("### Answer:")[0].strip()
        solution = query.split("### Answer:")[-1].strip()
        return question_part, solution

class KKProcessor:
    def __init__(self):
        self.answer_pattern = re.compile(r'\((\d+)\)\s*(.*?)\s*(?=\(\d+\)|$)', re.DOTALL)

    def parse_ground_truth(self, gt: str) -> List[str]:
        """解析参考答案"""
        return [s.strip() for s in gt.split("\n") if s.strip()]

    def parse_pred_answer(self, answer: str) -> List[str]:
        """解析预测答案"""
        matches = self.answer_pattern.findall(answer)
        return [f"({num}) {text.strip()}" for num, text in matches]

class RewardService: 
    def __init__(self, args):
        self._initialize_components(args)
        self._setup_metrics()
        self.kk_processor = KKProcessor()

    def _initialize_components(self, args):
        self.dataset = self._load_dataset(args.data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.reward_pretrain, 
            trust_remote_code=True,
        )
        self.log_file = args.log_file

    def _load_dataset(self, data_path: str) -> Dict:
        """Load dataset from JSONL file"""
        dataset = {}
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                dataset[item["quiz"]] = item["solution_text"]
        return dataset

    def _setup_metrics(self):
        self.metrics = {
            "total_processed": 0,
            "format_errors": 0,
            "answer_errors": 0
        }

    def calculate_rewards(self, queries: List[str]) -> List[List[float]]:
        processed_data = self._process_queries(queries)
        scores = self._evaluate_solutions(processed_data)
        self._record_results(processed_data, scores)
        return self._format_scores(scores)

    def _process_queries(self, queries: List[str]) -> List[Dict]:
        results = []
        for query in queries:
            question, solution = KKSolutionParser.parse_query(query)
            thought, pred_answer = KKSolutionParser.extract_components(solution)
            
            results.append({
                "raw_query": query,
                "question": question,
                "solution": solution,
                "pred_answer": pred_answer,
                "ref_answer": self.dataset.get(question, ""),
                "thought": thought,
                "format_correct": self._check_format(thought, pred_answer)
            })
        return results

    def _check_format(self, thought: str, answer: str) -> bool:
        """检查格式是否符合要求"""
        return bool(thought) and bool(answer) and ("<think>" in self._order_check(answer))

    def _order_check(self, text: str) -> str:
        """检查标签顺序"""
        think_pos = text.find("<think>")
        answer_pos = text.find("<answer>")
        if think_pos == -1 or answer_pos == -1:
            return "invalid"
        return "valid" if think_pos < answer_pos else "invalid"

    def _evaluate_solutions(self, data: List[Dict]) -> List[Dict]:
        for item in data:
            # 格式评分
            format_score = 1 if item["format_correct"] else -1
            
            # 答案评分
            gt_answers = self.kk_processor.parse_ground_truth(item["ref_answer"])
            pred_answers = self.kk_processor.parse_pred_answer(item["pred_answer"])
            
            answer_correct = set(gt_answers) == set(pred_answers)
            answer_score = 1 if answer_correct else -1
            
            item.update({
                "format_score": format_score,
                "answer_score": answer_score,
                "total_score": format_score + answer_score
            })
            
            # 更新统计
            self.metrics["total_processed"] += 1
            if not item["format_correct"]:
                self.metrics["format_errors"] += 1
            if not answer_correct:
                self.metrics["answer_errors"] += 1
        
        logger.info(f"Processing Metrics: {self.metrics}")
        return data

    def _record_results(self, data: List[Dict], scores: List[Dict]):
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as f:
            for item in data:
                record = {
                    "question": item["question"],
                    "thought": item["thought"],
                    "pred_answer": item["pred_answer"],
                    "ref_answer": item["ref_answer"],
                    "format_score": item["format_score"],
                    "answer_score": item["answer_score"],
                    "total_score": item["total_score"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _format_scores(self, data: List[Dict]) -> List[List[float]]:
        return [[item["total_score"]] for item in data]

class APIServer:
    def __init__(self, args):
        self.app = FastAPI()
        self.args = args
        self.service = RewardService(args)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/get_reward")
        async def get_reward(request: Request):
            payload = await request.json()
            queries = payload.get("query", [])
            rewards = self.service.calculate_rewards(queries)
            logger.info(f"Processed {len(queries)} queries")
            return JSONResponse({"rewards": rewards})

    def run(self):
        uvicorn.run(
            self.app,
            host=self.args.host,
            port=self.args.port,
            log_level="info"
        )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KK Reward Model Service")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to KK dataset")
    parser.add_argument("--reward_pretrain", type=str, required=True,
                       help="Pretrained model path")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                       help="Service port number")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                       help="Service host address")
    parser.add_argument("--log_file", type=str, 
                       help="Path to JSONL log file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    server = APIServer(args)
    server.run()
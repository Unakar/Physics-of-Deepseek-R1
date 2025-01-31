import random
import re

# 定义系统提示和XML格式
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# 提取XML中的答案部分
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# 提取标准答案（#### 后的部分）
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def generate_random_question_and_answer():
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    operation = random.choice(['+', '-', '*'])
    
    if operation == '+':
        question = f"What is {num1} + {num2}?"
        answer = num1 + num2
    elif operation == '-':
        question = f"What is {num1} - {num2}?"
        answer = num1 - num2
    elif operation == '*':
        question = f"What is {num1} * {num2}?"
        answer = num1 * num2
    
    # 标准答案格式为 #### 答案
    standard_answer = f"#### {answer}"
    
    # 模拟模型的回答
    reasoning = f"{num1} {operation} {num2} equals {answer}."
    model_response = XML_COT_FORMAT.format(reasoning=reasoning, answer=str(answer))
    
    return {
        'question': question,
        'standard_answer': standard_answer,
        'model_response': model_response
    }

# 奖励函数：正确性奖励
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# 奖励函数：检查答案是否为数字
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# 奖励函数：严格格式检查
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

# 奖励函数：宽松格式检查
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

# 奖励函数：统计XML标签数量
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# 主程序
if __name__ == "__main__":
    # 随机生成一条数据
    data = generate_random_question_and_answer()
    
    print("Generated Question:", data['question'])
    print("Standard Answer:", data['standard_answer'])
    print("Model Response:", data['model_response'])
    
    prompts = [[{'role': 'user', 'content': data['question']}]]
    completions = [[{'content': data['model_response']}]]
    answer = [extract_hash_answer(data['standard_answer'])]
    

    print("\nCorrectness Reward:", correctness_reward_func(prompts, completions, answer))
    print("Integer Reward:", int_reward_func(completions))
    print("Strict Format Reward:", strict_format_reward_func(completions))
    print("Soft Format Reward:", soft_format_reward_func(completions))
    print("XML Count Reward:", xmlcount_reward_func(completions))
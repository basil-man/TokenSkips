import json
import re

def load_test_questions(path):
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                questions.append(record["question"].strip())
    return questions

def clean_text(text):
    # 正则匹配形如 "<|eot_id|>数字(.数字)?<|eot_id|>" 的子串，并将其删除
    pattern = re.compile(r"<\|eot_id\|>\d+(\.\d+)?<\|eot_id\|>")
    return pattern.sub("", text).strip()

def load_dataset_inputs(path):
    inputs = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            if "input" in entry:
                content = entry["input"].strip()
            elif "output" in entry:
                content = entry["output"].strip()
            else:
                continue
            # 清理内容中符合条件的部分
            content = clean_text(content)
            inputs.append(content)
    return inputs

def main():
    test_path = "/home/featurize/work/TokenSkip/datasets/gsm8k/test.jsonl"
    dataset_path = "/home/featurize/work/TokenSkip/data/mydataset_compressed_gsm8k_llmlingua2_qwen_7B.json"
    
    test_questions = load_test_questions(test_path)
    dataset_inputs = load_dataset_inputs(dataset_path)
    
    duplicate_questions = []
    for question in test_questions:
        # 清理测试题问句两边的空白字符
        q_clean = question.strip()
        if q_clean in dataset_inputs:
            duplicate_questions.append(q_clean)
    
    if duplicate_questions:
        print("发现以下重复的题目：")
        for q in duplicate_questions:
            print(q)
    else:
        print("未发现重复的题目。")

if __name__ == "__main__":
    main()
import os
import json
import random
import numpy as np
from transformers import AutoTokenizer  # 新增

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  # 加载tokenizer
MAX_TOKENS = 11000
HALF_MAX = MAX_TOKENS // 2

def load_json(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data

def write_list_to_json(list, file_path):
    with open(file_path, 'w') as f:
        json.dump(list, f, ensure_ascii=False, indent=1)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_all_data(input_dir="outputs/"):
    original_data = load_json(os.path.join(input_dir, "predictions_formatted.jsonl"))
    compressed_data_0 = load_json(os.path.join(input_dir, "Compression/train_outputs_compressed_ratio_0.9.jsonl"))
    compressed_data_1 = load_json(os.path.join(input_dir, "Compression/train_outputs_compressed_ratio_0.8.jsonl"))
    compressed_data_2 = load_json(os.path.join(input_dir, "Compression/train_outputs_compressed_ratio_0.7.jsonl"))
    compressed_data_3 = load_json(os.path.join(input_dir, "Compression/train_outputs_compressed_ratio_0.6.jsonl"))
    compressed_data_4 = load_json(os.path.join(input_dir, "Compression/train_outputs_compressed_ratio_0.5.jsonl"))
    return [original_data, compressed_data_0, compressed_data_1, compressed_data_2, compressed_data_3, compressed_data_4]

def truncate_input(input_text):
    tokens = tokenizer.encode(input_text, add_special_tokens=False)
    if len(tokens) <= MAX_TOKENS:
        return input_text
    truncated_tokens = tokens[:HALF_MAX] + tokens[-HALF_MAX:]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def get_llamafactory_input():
    compressed_data_list = load_all_data()
    original_data = compressed_data_list[0]
    compression_ratio_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    datalines = []
    for i in range(len(original_data)):
        data_index = random.choice([0, 1, 2, 3, 4, 5])
        if data_index == 0:
            input_data = original_data[i]['prompt']
            answer = original_data[i]['answer']
            cot = original_data[i]['response_cot']
            output_data = f"{cot}"
        else:
            compression_ratio = compression_ratio_list[data_index]
            compressed_data = compressed_data_list[data_index]
            input_data = compressed_data[i]['input']
            answer = compressed_data[i]['model_answer']
            cot = compressed_data[i]['compressed_cot']
            output_data = f"{cot}"
        redundant_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease read the following text and answer the questions below.\n"
        if input_data.strip().startswith(redundant_prompt):
            input_data = input_data.strip()[len(redundant_prompt):].lstrip()

        # ⚠️ 处理token数超过限制的情况
        input_data = truncate_input(input_data)

        data = {
            "instruction": f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease read the following text and answer the questions below.\n",
            "input": f'{input_data}\n<|eot_id|>{compression_ratio}<|eot_id|><|im_end|>\n<|im_start|>assistant\n',
            "output": output_data
        }
        datalines.append(data)

    print(len(datalines))
    random.shuffle(datalines)
    write_list_to_json(datalines, './outputs/mydataset_compressed_longbench_llmlingua_qwen_7B.json')

if __name__ == '__main__':
    seed_everything(42)
    get_llamafactory_input()

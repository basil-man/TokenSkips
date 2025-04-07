import os, csv, json
import argparse
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import re
from copy import deepcopy
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入generate_completions函数
import sys
sys.path.append('/home/featurize/work/TokenSkip')
from eval.utils import generate_completions

# 读取配置文件
model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

# 读取各个 prompt 模板
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def query_llm(prompt, model_name, tokenizer, temperature=0.0, max_new_tokens=128, cot_flag=False, model=None, adapter_path=None, use_adapter=False):
    max_len = maxlen_map[model_name]
    max_input_tokens = max_len - max_new_tokens
    
    # 处理输入长度限制
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if input_ids.shape[1] > max_input_tokens:
        half = max_input_tokens // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -(max_input_tokens - half):]], dim=1)
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # 如果没有传入模型，则加载模型
    if model is None:
        print("Loading model and tokenizer...")
        model_path = model_map[model_name] if model_name in model_map else model_name
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        
        if use_adapter and adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, device_map="auto")
            model = model.merge_and_unload()
    
    # 设置padding和pad token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 设置停止序列
    stop_id_sequences = []
    if tokenizer.eos_token_id is not None:
        stop_id_sequences = [[tokenizer.eos_token_id]]
    
    # 使用generate_completions函数生成回答
    torch.cuda.synchronize()
    start_time = time.time()
    outputs, _ = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        top_p=1.0,
        batch_size=1,
        stop_id_sequences=stop_id_sequences if stop_id_sequences else None,
        end_of_generation_id_sequence=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    )
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    if len(outputs) > 0:
        response_text = outputs[0]
        # 计算token数量
        if cot_flag:
            response_tokens = tokenizer(response_text, return_tensors="pt")['input_ids'].shape[1]
        else:
            response_tokens = tokenizer(prompt + response_text, return_tensors="pt")['input_ids'].shape[1]
        return response_text, response_tokens
    else:
        return '', 0

# 其他代码保持不变

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None
def get_pred(data, args, out_file, tokenizer, model):
    # 模型和tokenizer已经在main函数中加载，直接使用传入的参数

    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot

        prompt = template.replace('$DOC$', context.strip()) \
                         .replace('$Q$', item['question'].strip()) \
                         .replace('$C_A$', item['choice_A'].strip()) \
                         .replace('$C_B$', item['choice_B'].strip()) \
                         .replace('$C_C$', item['choice_C'].strip()) \
                         .replace('$C_D$', item['choice_D'].strip())
        print(prompt[:100])
        # 如果提供了ratio参数，在prompt末尾添加压缩率标记
        if args.ratio is not None:
            prompt = f"{prompt}\n<|eot_id|>{args.ratio}<|eot_id|>"
        
        prompt_1=prompt
        output, token_count = query_llm(prompt, args.model, tokenizer, temperature=0.0, max_new_tokens=10000, cot_flag=True, model=model, adapter_path=args.adapter_path, use_adapter=args.use_adapter)
        if output == '':
            continue

        if args.cot:
            response_cot = output.strip()
            item['response_cot'] = response_cot
            item['cot_token_count'] = token_count  # 记录COT的token数

            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()) \
                                           .replace('$Q$', item['question'].strip()) \
                                           .replace('$C_A$', item['choice_A'].strip()) \
                                           .replace('$C_B$', item['choice_B'].strip()) \
                                           .replace('$C_C$', item['choice_C'].strip()) \
                                           .replace('$C_D$', item['choice_D'].strip()) \
                                           .replace('$COT$', response_cot)

            output, token_count = query_llm(prompt, args.model, tokenizer, temperature=0.0, max_new_tokens=10000, cot_flag=True, model=model, adapter_path=args.adapter_path, use_adapter=args.use_adapter)
            if output == '':
                continue

        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context
        item['final_token_count'] = token_count  # 记录最终答案的token数
        item['prompt'] = prompt_1
        # 移除进程锁，直接写入文件
        with open(out_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    
    # 根据模式添加后缀
    mode_suffix = f"_{args.mode}"
    
    # 添加压缩率后缀（如果提供了ratio参数）
    ratio_suffix = f"_{args.ratio}" if args.ratio is not None else ""
    
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}{ratio_suffix}{mode_suffix}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_no_context{ratio_suffix}{mode_suffix}.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_cot{ratio_suffix}{mode_suffix}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"{ratio_suffix}{mode_suffix}.jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    data_all = [{
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "answer": item["answer"],
        "context": item["context"]
    } for item in dataset]
    
    # 同时根据difficulty和length字段进行分层随机抽样，保持8:2的比例划分数据集
    import random
    random.seed(42)  # 设置随机种子，确保划分结果可复现
    
    # 按照(difficulty, length)组合键分组
    combined_groups = {}
    for item in data_all:
        difficulty = item['difficulty']
        length = item['length']
        combined_key = (difficulty, length)
        if combined_key not in combined_groups:
            combined_groups[combined_key] = []
        combined_groups[combined_key].append(item)
    
    # 对每个组合分别进行随机打乱和8:2划分
    train_data = []
    test_data = []
    
    for combined_key, items in combined_groups.items():
        random.shuffle(items)  # 随机打乱每个组的数据
        split_idx = int(len(items) * 0.8)  # 80%的数据用于训练
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])
    
    print(f"总数据：{len(data_all)}条")
    print(f"组合分组数：{len(combined_groups)}个")
    
    # 根据模式选择相应的数据子集
    if args.mode == "train":
        data_all = train_data
        print(f"使用训练集: {len(data_all)}条数据")
        
        # 打印训练集中各个维度的分布情况
        difficulty_counts = {}
        length_counts = {}
        for item in data_all:
            difficulty = item['difficulty']
            length = item['length']
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            length_counts[length] = length_counts.get(length, 0) + 1
        
        print(f"训练集难度分布: {difficulty_counts}")
        print(f"训练集长度分布: {length_counts}")
        
    else:  # test模式
        data_all = test_data
        print(f"使用测试集: {len(data_all)}条数据")
        
        # 打印测试集中各个维度的分布情况
        difficulty_counts = {}
        length_counts = {}
        for item in data_all:
            difficulty = item['difficulty']
            length = item['length']
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            length_counts[length] = length_counts.get(length, 0) + 1
        
        print(f"测试集难度分布: {difficulty_counts}")
        print(f"测试集长度分布: {length_counts}")

    # 缓存已处理的数据，避免重复调用
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    data = [item for item in data_all if item["_id"] not in has_data]
    
    # 加载tokenizer和模型（只加载一次）
    print("Loading model and tokenizer...")
    model_path = model_map[args.model] if args.model in model_map else args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    if args.use_adapter and args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, device_map="auto")
        model = model.merge_and_unload()
    
    # 取消多进程处理，直接处理所有数据
    get_pred(data, args, out_file, tokenizer, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true', help="使用链式思考（COT）")
    parser.add_argument("--no_context", "-nc", action='store_true', help="不使用上下文，直接测记忆")
    parser.add_argument("--rag", "-rag", type=int, default=0, help="若不使用RAG则为0，否则为使用top-N检索的N值")
    parser.add_argument("--n_proc", "-n", type=int, default=16, help="使用的进程数")
    parser.add_argument("--mode", "-md", type=str, choices=["train", "test"], default="train", help="指定运行模式：训练(train)或测试(test)")
    parser.add_argument("--ratio", "-r", type=float, default=None, help="压缩率，当提供时会在prompt末尾添加<|eot_id|>{compression_ratio}<|eot_id|>")
    parser.add_argument("--adapter_path", type=str, default=None, help="adapter路径，用于TokenSkip")
    parser.add_argument("--use_adapter", action='store_true', default=False, help="是否使用adapter")
    parser.add_argument("--lazy_load", action='store_true', default=False, help="是否在每个进程中单独加载模型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_new_tokens", type=int, default=10000, help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--auto_gamma", action='store_true', default=False, help="是否启用自动gamma")
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    main()

import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
import openai
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

# 读取配置文件
model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

# 配置请求参数，与curl一致
API_BASE = "http://127.0.0.1:8001/v1"
API_KEY = "token-abc123"
client = openai.OpenAI(base_url=API_BASE, api_key=API_KEY)  # 使用新的 OpenAI 客户端

# 读取各个 prompt 模板
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model, tokenizer, temperature=0.0, max_new_tokens=128, cot_flag=False):
    max_len = maxlen_map[model]
    max_input_tokens = max_len - max_new_tokens
    
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_input_tokens:
            half = max_input_tokens // 2
            input_ids = input_ids[:half] + input_ids[-(max_input_tokens - half):]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_input_tokens:
            half = max_input_tokens // 2
            input_ids = input_ids[:half] + input_ids[-(max_input_tokens - half):]
            prompt = tokenizer.decode(input_ids)

    tries = 0
    if model in model_map:
        model = model_map[model]

    while tries < 5:
        tries += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            response_text = response.choices[0].message.content
            if cot_flag:
                response_tokens = response.usage.completion_tokens
            else:
                response_tokens = response.usage.total_tokens
            return response_text, response_tokens
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"Error Occurs: \"{str(e)}\" Retry ...")
            time.sleep(1)
    else:
        print("Max tries. Failed.")
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
def get_pred(data, args, out_file, lock):
    if "gpt" in args.model or "o1" in args.model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)

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
        output, token_count = query_llm(prompt, args.model, tokenizer, temperature=0.0, max_new_tokens=10000, cot_flag=True)
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

            output, token_count = query_llm(prompt, args.model, tokenizer, temperature=0.0, max_new_tokens=10000, cot_flag=True)
            if output == '':
                continue

        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context
        item['final_token_count'] = token_count  # 记录最终答案的token数
        item['prompt'] = prompt_1
        with lock:
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


    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    lock = mp.Lock()  # 进程锁，用于同步写入
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, out_file, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

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
    args = parser.parse_args()
    main()

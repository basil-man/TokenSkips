import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class PromptCompressor:
    def __init__(self, model_name="./model/Qwen2.5-Math-7B-Instruct"):
        """初始化PromptCompressor
        
        Args:
            model_name: Qwen2.5-Math-7B-Instruct模型路径
        """
        print(f"Loading model from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
    
    def _calculate_token_importance(self, input_ids, attention_mask=None):
        """计算每个token的重要性分数
        
        使用模型的输出logits计算token的重要性
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 获取输出的logits
            logits = outputs.logits
            
            # 计算每个位置预测下一个token的confidence
            next_token_probs = torch.softmax(logits, dim=-1)
            # 取每个位置预测概率最高的token的概率值
            highest_probs = torch.max(next_token_probs, dim=-1).values
            
            # 重要性分数 - 使用预测概率作为基础，位置越靠后的token影响越大
            seq_len = input_ids.size(1)
            position_weights = torch.linspace(0.5, 1.0, seq_len).to(input_ids.device)
            importance_scores = highest_probs * position_weights
            
            return importance_scores.cpu().numpy()[0]  # 只取第一个样本，并转为numpy数组
    
    def compress_prompt(self, prompt, rate=0.5, force_tokens=None, force_reserve_digit=False, drop_consecutive=False):
        """压缩提示文本
        
        Args:
            prompt: 要压缩的文本
            rate: 目标压缩率 (保留的token比例)
            force_tokens: 强制保留的token列表
            force_reserve_digit: 是否强制保留数字
            drop_consecutive: 是否允许连续删除token
            
        Returns:
            dict: 包含压缩结果的字典
        """
        # 对输入文本进行tokenize
        tokenized = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        
        # 计算token的重要性分数
        importance_scores = self._calculate_token_importance(input_ids, attention_mask)
        
        # 获取原始tokens
        tokens = [self.tokenizer.decode([id]) for id in input_ids[0].cpu().numpy()]
        
        # 标记需要强制保留的tokens
        keep_mask = np.zeros(len(tokens), dtype=bool)
        
        # 保留第一个和最后一个token
        keep_mask[0] = True
        keep_mask[-1] = True
        
        # 处理强制保留的token
        if force_tokens:
            for i, token in enumerate(tokens):
                if any(ft in token for ft in force_tokens):
                    keep_mask[i] = True
        
        # 处理强制保留数字
        if force_reserve_digit:
            for i, token in enumerate(tokens):
                if any(c.isdigit() for c in token):
                    keep_mask[i] = True
        
        # 计算需要保留的token数量
        n_keep = max(2, int(rate * len(tokens)))
        n_remove = len(tokens) - n_keep
        
        # 如果需要删除的token数小于等于0，直接返回原始提示
        if n_remove <= 0:
            return {
                "compressed_prompt": prompt,
                "origin_tokens": len(tokens),
                "compressed_tokens": len(tokens),
                "rate": 1.0
            }
        
        # 根据重要性分数排序（不考虑已强制保留的token）
        candidates_to_remove = np.where(~keep_mask)[0]
        importance_to_remove = importance_scores[candidates_to_remove]
        sorted_idx = np.argsort(importance_to_remove)
        
        # 从低重要性到高重要性排序token索引
        remove_idx = candidates_to_remove[sorted_idx][:n_remove]
        remove_idx = np.sort(remove_idx)
        
        # 如果启用了drop_consecutive参数，避免连续删除
        if drop_consecutive:
            valid_remove_idx = []
            last_removed = -2
            for idx in remove_idx:
                if idx != last_removed + 1:
                    valid_remove_idx.append(idx)
                    last_removed = idx
            remove_idx = np.array(valid_remove_idx)
        
        # 生成最终保留mask
        final_keep_mask = np.ones(len(tokens), dtype=bool)
        final_keep_mask[remove_idx] = False
        
        # 构建压缩后的文本
        compressed_tokens = [token for i, token in enumerate(tokens) if final_keep_mask[i]]
        compressed_prompt = self.tokenizer.convert_tokens_to_string(compressed_tokens)
        
        # 统计压缩结果
        compressed_token_count = len(self.tokenizer.encode(compressed_prompt))
        original_token_count = len(self.tokenizer.encode(prompt))
        actual_rate = compressed_token_count / original_token_count
        
        return {
            "compressed_prompt": compressed_prompt,
            "origin_tokens": original_token_count,
            "compressed_tokens": compressed_token_count,
            "rate": actual_rate
        }

def load_jsonl(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data

def save_jsonl(data, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    for item in data:
        with open(output_path, 'a+', encoding='utf-8') as f:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

def filter_correct_outputs(input_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions.jsonl",
                           output_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_correct.jsonl"):
    data = load_jsonl(input_path)
    correct_data = []
    for i in range(len(data)):
        if data[i]['accuracy']:
            correct_data.append(data[i])
    print(f"Original Samples: {len(data)}, Correct Samples: {len(correct_data)}, Accuracy: {len(correct_data) / len(data)}")
    save_jsonl(correct_data, output_path)

def filter_formatted_outputs(input_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_correct.jsonl",
                             output_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_formatted.jsonl", 
                             model_type="qwen"):
    data = load_jsonl(input_path)
    formatted_data = []
    for i in range(len(data)):
        if data[i]['cot_length'] > 500:
            continue
        if model_type == "llama3":
            spans = data[i]["output"].split('\n\nThe final answer is:')
            if len(spans) == 2:
                data[i]["cot"] = spans[0]
                formatted_data.append(data[i])
        elif model_type == "qwen":
            formatted_data.append(data[i])
        else:
            raise ValueError(f"Model Type {model_type} is not supported.")
    print(f"Original Samples: {len(data)}, Formatted Samples: {len(formatted_data)}")
    save_jsonl(formatted_data, output_path)
# 修改 compress_with_algorithm 函数，新增 compressor 参数
def compress_with_algorithm(data, compression_ratio=0.5, model_type="qwen",
                            model_path="./model/Qwen2.5-Math-7B-Instruct", compressor=None):
    """
    使用Qwen2.5-Math-7B-Instruct模型通过算法压缩CoT输出
    """
    if model_type == "llama3":
        cot_type = "cot"
    elif model_type == "qwen":
        cot_type = "model_output"
    else:
        raise ValueError(f"Model Type {model_type} is not supported.")
    
    # 如果未传入 compressor，则初始化新的
    if compressor is None:
        compressor = PromptCompressor(model_name=model_path)
    
    compressed_data = []
    
    for i in tqdm(range(len(data))):
        cot_output = data[i][cot_type]
        
        # 使用算法压缩文本
        if model_type == "llama3":
            compress_result = compressor.compress_prompt(
                cot_output, 
                rate=compression_ratio,
                force_tokens=['Step', ':'],
                force_reserve_digit=True,
                drop_consecutive=True
            )
        else:
            compress_result = compressor.compress_prompt(
                cot_output, 
                rate=compression_ratio
            )
        
        compressed_data_line = {
            'question': data[i]['messages'][0]['content'],
            'input': data[i]['prompt'],
            'output': data[i]['model_output'],
            'answer': data[i]['answer'],
            'model_answer': data[i]['prediction'],
            'is_correct': data[i]['accuracy'],
            'cot': data[i][cot_type],
            'compressed_cot': compress_result['compressed_prompt'],
            'original_cot_tokens': compress_result['origin_tokens'],
            'compressed_cot_tokens': compress_result['compressed_tokens'],
            'compression_rate': compress_result['rate']
        }
        compressed_data.append(compressed_data_line)
    
    return compressed_data
# 修改 compress_cot_outputs 函数，在循环前复用同一个 PromptCompressor 实例
def compress_cot_outputs(input_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_formatted.jsonl",
                         output_dir="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Compression_Qwen", 
                         model_type="qwen",
                         model_path="./model/Qwen2.5-Math-7B-Instruct"):
    """
    使用算法压缩CoT输出，尝试不同压缩率
    """
    data = load_jsonl(input_path)
    ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    # 在循环外创建 PromptCompressor 实例，复用模型加载
    compressor = PromptCompressor(model_name=model_path)
    
    for compression_ratio in ratio_list:
        output_path = os.path.join(output_dir, f"train_outputs_compressed_ratio_{compression_ratio}.jsonl")
        compressed_data = compress_with_algorithm(data, compression_ratio=compression_ratio, model_type=model_type, model_path=model_path, compressor=compressor)
        save_jsonl(compressed_data, output_path)
        get_average_compress_rate(compressed_data)
        
def get_average_compress_rate(data):
    compress_rate = 0
    for i in range(len(data)):
        compress_rate += data[i]['compressed_cot_tokens'] / data[i]['original_cot_tokens']
    compress_rate = compress_rate / len(data)
    print(f"Average Compression Rate: {compress_rate}")

def data_processing_gsm8k(input_dir="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/", 
                         model_type="qwen",
                         model_path="./model/Qwen2.5-Math-7B-Instruct"):
    """
    GSM8K数据处理的整体流程
    """
    input_path = os.path.join(input_dir, "Original/train/samples/predictions.jsonl")
    correct_path = os.path.join(input_dir, "Original/train/samples/predictions_correct.jsonl")
    formatted_path = os.path.join(input_dir, "Original/train/samples/predictions_formatted.jsonl")
    compressed_dir = os.path.join(input_dir, "Compression")

    filter_correct_outputs(input_path=input_path, output_path=correct_path)
    filter_formatted_outputs(input_path=correct_path, output_path=formatted_path, model_type=model_type)
    compress_cot_outputs(input_path=formatted_path, output_dir=compressed_dir, model_type=model_type, model_path=model_path)

if __name__ == '__main__':
    data_processing_gsm8k()
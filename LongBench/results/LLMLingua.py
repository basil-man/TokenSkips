import os
import json
from tqdm import tqdm
from llmlingua import PromptCompressor


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
    """
    筛选正确输出的样本。这里使用新数据中的 judge 字段判断正确性。
    """
    data = load_jsonl(input_path)
    correct_data = []
    for i in range(len(data)):
        # 假设 judge 字段为布尔型或能作为判断依据（例如 "correct" 字符串也可以根据需求转换）
        if data[i]['judge']:
            correct_data.append(data[i])
    print(f"原始样本数: {len(data)}, 正确样本数: {len(correct_data)}, 准确率: {len(correct_data) / len(data):.2f}")
    save_jsonl(correct_data, output_path)

def filter_formatted_outputs(input_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_correct.jsonl",
                             output_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_formatted.jsonl", 
                             model_type="qwen"):
    """
    筛选格式化输出的样本。基于新数据中的 response_cot 与 cot_token_count 字段处理链式思考部分。
    """
    data = load_jsonl(input_path)
    formatted_data = []
    for i in range(len(data)):
        # 如果链式思考的 token 数超过 500，则跳过
        if data[i]['cot_token_count'] > 1200000:
            continue
        if model_type == "llama3":
            # 如果模型为 llama3，则尝试通过分隔符提取链式思考部分
            spans = data[i]["response_cot"].split('\n\nThe final answer is:')
            if len(spans) == 2:
                data[i]["response_cot"] = spans[0]
                formatted_data.append(data[i])
        elif model_type == "qwen":
            # 对于 qwen 模型，直接使用 response_cot
            formatted_data.append(data[i])
        else:
            raise ValueError(f"Model Type {model_type} is not supported.")
    print(f"原始样本数: {len(data)}, 格式化样本数: {len(formatted_data)}")
    save_jsonl(formatted_data, output_path)

def LLMLingua(data, compression_ratio=0.5, model_type="qwen",
              llmlingua_path="/your_model_path/llmlingua-2-xlm-roberta-large-meetingbank"):
    """
    使用 LLMLingua-2 压缩链式思考输出，这里统一使用 response_cot 字段作为链式思考。
    """
    # 新数据中统一使用 response_cot 作为链式思考，不再区分模型类型
    cot_field = "response_cot"

    llm_lingua = PromptCompressor(
        model_name=llmlingua_path,
        use_llmlingua2=True,  # 是否使用 llmlingua-2
    )
    compressed_data = []
    for i in tqdm(range(len(data))):
        cot_output = data[i][cot_field]
        # 根据模型类型可能设置不同的压缩参数
        if model_type == "llama3":
            compressed_prompt = llm_lingua.compress_prompt(cot_output, rate=compression_ratio, force_tokens=['Step', ':'], force_reserve_digit=True, drop_consecutive=True)
        elif model_type == "qwen":
            compressed_prompt = llm_lingua.compress_prompt(cot_output, rate=compression_ratio)
        else:
            raise ValueError(f"Model Type {model_type} is not supported.")
        compressed_data_line = {
            'question': data[i]['question'],
            'input': data[i]['prompt'],
            'output': data[i]['response'],
            'answer': data[i]['answer'],
            'model_answer': data[i]['pred'],
            'is_correct': data[i]['judge'],
            'cot': data[i][cot_field],
            'compressed_cot': compressed_prompt['compressed_prompt'],
            'original_cot_tokens': compressed_prompt['origin_tokens'],
            'compressed_cot_tokens': compressed_prompt['compressed_tokens'],
            'compression_rate': compressed_prompt['rate']
        }
        compressed_data.append(compressed_data_line)
    return compressed_data

def compress_cot_outputs(input_path="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Original/samples/predictions_formatted.jsonl",
                         output_dir="outputs/Qwen2.5-7B-Instruct/gsm8k/7b/Compression", model_type="qwen",
                         llmlingua_path="llmlingua-2-xlm-roberta-large-meetingbank"):
    """
    针对不同压缩比例，使用 LLMLingua-2 压缩链式思考输出。
    """
    data = load_jsonl(input_path)
    ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5]
    for compression_ratio in ratio_list:
        output_path = os.path.join(output_dir, f"train_outputs_compressed_ratio_{compression_ratio}.jsonl")
        compressed_data = LLMLingua(data, compression_ratio=compression_ratio, model_type=model_type, llmlingua_path=llmlingua_path)
        save_jsonl(compressed_data, output_path)
        get_average_compress_rate(compressed_data)

def get_average_compress_rate(data):
    compress_rate = 0
    for i in range(len(data)):
        compress_rate += data[i]['compressed_cot_tokens'] / data[i]['original_cot_tokens']
    compress_rate = compress_rate / len(data)
    print(f"平均压缩率: {compress_rate:.2f}")

def data_processing_gsm8k(input_dir="outputs/", model_type="qwen",
                          llmlingua_path="./model/llmlingua-2-xlm-roberta-large-meetingbank"):
    """
    GSM8K 数据处理的整体流水线，基于新数据输入的字段。
    """
    input_path = os.path.join(input_dir, "Qwen2.5-7B-Instruct_cot_train.jsonl")
    correct_path = os.path.join(input_dir, "predictions_correct.jsonl")
    formatted_path = os.path.join(input_dir, "predictions_formatted.jsonl")
    compressed_dir = os.path.join(input_dir, "Compression")

    filter_correct_outputs(input_path=input_path, output_path=correct_path)
    filter_formatted_outputs(input_path=correct_path, output_path=formatted_path, model_type=model_type)
    compress_cot_outputs(input_path=formatted_path, output_dir=compressed_dir, model_type=model_type, llmlingua_path=llmlingua_path)

if __name__ == '__main__':
    data_processing_gsm8k()

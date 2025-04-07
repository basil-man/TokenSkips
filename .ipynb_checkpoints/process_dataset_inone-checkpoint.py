import json
import os
import math
import argparse
import random

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="处理数据集并分割GSM8K数据集")
parser.add_argument("--ratio", "-r", type=float, default=0.1, help="要处理的数据比例 (0-1之间的浮点数，默认为0.1)")
parser.add_argument("--shuffle", "-s", action="store_true", help="是否打乱数据集")
parser.add_argument("--seed", type=int, default=42, help="随机数种子，用于数据打乱")
args = parser.parse_args()

# 获取参数值
AUTO_RATIO = args.ratio
SHUFFLE = args.shuffle
SEED = args.seed

# 如果需要打乱，设置随机种子以确保可重复性
if SHUFFLE:
    random.seed(SEED)
    print(f"数据将被打乱，使用随机种子: {SEED}")

def process_dataset():
    """处理数据集，提取指定比例的数据"""
    print("\n--- 开始处理数据集 ---")
    # 定义路径
    input_file = "./outputs/mydataset_reordered.json"
    output_dir = "./data"
    output_file = os.path.join(output_dir, "mydataset_compressed_gsm8k_llmlingua2_qwen_7B.json")

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 检查并删除已存在的输出文件
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除原有文件: {output_file}")

    # 读取输入文件
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # 如果需要打乱数据
    if SHUFFLE:
        random.seed(SEED)
        random.shuffle(data)
        print(f"已打乱数据集，共 {len(data)} 条记录")
        
    print(data[:2])
    # 计算要保留的数据量 (使用 1-AUTO_RATIO)
    num_to_keep = math.ceil(len(data) * (1 - AUTO_RATIO))
    subset_data = data[:num_to_keep]

    # 写入输出文件
    with open(output_file, "w") as f:
        json.dump(subset_data, f, indent=1)

    print(f"成功将前{num_to_keep}条数据（{(1-AUTO_RATIO) * 100}%）写入到 {output_file}")

def split_gsm8k():
    """将GSM8K数据集按照AUTO_RATIO比例分割"""
    print("\n--- 开始分割GSM8K数据集 ---")
    # 定义路径
    input_file = "./original_datasets/gsm8k/train_filtered.jsonl"
    output_dir = "./datasets/gsm8k"
    train_output_file = os.path.join(output_dir, "train.jsonl")
    auto_output_file = os.path.join(output_dir, "test.jsonl")

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(train_output_file):
        os.remove(train_output_file)
        print(f"已删除原有文件: {train_output_file}")
    if os.path.exists(auto_output_file):
        os.remove(auto_output_file)
        print(f"已删除原有文件: {auto_output_file}")
    
    # 读取输入文件
    data = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # 如果需要打乱数据
    if SHUFFLE:
        # 使用与第一个数据集相同的随机种子进行打乱
        random.seed(SEED)
        random.shuffle(data)
      
        print(f"已打乱GSM8K数据集，共 {len(data)} 条记录")
    print(data[:2])
    # 计算分割点
    split_point = math.ceil(len(data) * (1 - AUTO_RATIO))

    # 分割数据
    train_data = data[:split_point]
    auto_data = data[split_point:]

    # 写入训练数据文件
    with open(train_output_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # 写入自动数据文件
    with open(auto_output_file, "w") as f:
        for item in auto_data:
            f.write(json.dumps(item) + "\n")

    print(f"原始数据: {len(data)} 条")
    print(f"已将前{1-AUTO_RATIO:.1%}的数据 ({len(train_data)}条) 保存到: {train_output_file}")
    print(f"已将后{AUTO_RATIO:.1%}的数据 ({len(auto_data)}条) 保存到: {auto_output_file}")

if __name__ == "__main__":
    print(f"使用数据比例: {AUTO_RATIO}")
    if SHUFFLE:
        print("启用数据打乱")
    process_dataset()
    split_gsm8k()
    print("\n所有处理完成!")
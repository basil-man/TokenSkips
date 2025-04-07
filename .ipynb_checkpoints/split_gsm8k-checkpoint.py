import json
import os
import math
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="将GSM8K数据集按照AUTO_RATIO比例分割")
parser.add_argument("--ratio", "-r", type=float, default=0.1, help="要分割的数据比例 (0-1之间的浮点数，默认为0.1)")
args = parser.parse_args()

# 获取AUTO_RATIO的值
AUTO_RATIO = args.ratio

# 定义路径
input_file = "./original_datasets/gsm8k/train_filtered.jsonl"
output_dir = "./datasets/gsm8k"
train_output_file = os.path.join(output_dir, "train.jsonl")
auto_output_file = os.path.join(output_dir, "auto.jsonl")

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

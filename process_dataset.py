import json
import os
import math
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="处理数据集，提取指定比例的数据")
parser.add_argument("--ratio", "-r", type=float, default=0.1, help="要跳过的数据比例 (0-1 之间的浮点数，默认为 0.1)")
args = parser.parse_args()

# 获取AUTO_RATIO的值
AUTO_RATIO = args.ratio

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

# 计算要保留的数据量 (使用 1-AUTO_RATIO)
num_to_keep = math.ceil(len(data) * (1 - AUTO_RATIO))
subset_data = data[:num_to_keep]

# 写入输出文件
with open(output_file, "w") as f:
    json.dump(subset_data, f, indent=1)

print(f"成功将前{num_to_keep}条数据（{(1-AUTO_RATIO) * 100}%）写入到 {output_file}")

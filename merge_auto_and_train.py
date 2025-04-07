import json
import os
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="合并两个数据集并添加必要字段")
parser.add_argument("--first-dataset", default="new_dataset.json", help="第一个数据集文件路径 (默认: new_dataset.json)")
parser.add_argument(
    "--second-dataset",
    default="mydataset_compressed_gsm8k_llmlingua2_qwen_7B.json",
    help="第二个数据集文件路径 (默认: mydataset_compressed_gsm8k_llmlingua2_qwen_7B.json)",
)
parser.add_argument("--output", default="merged_dataset.json", help="输出文件路径 (默认: merged_dataset.json)")
args = parser.parse_args()

# 获取当前目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取两个数据集
first_dataset_path = args.first_dataset
if not os.path.isabs(first_dataset_path):
    first_dataset_path = os.path.join(current_dir, first_dataset_path)
with open(first_dataset_path, "r") as f:
    new_dataset = json.load(f)

second_dataset_path = args.second_dataset
if not os.path.isabs(second_dataset_path):
    second_dataset_path = os.path.join(current_dir, second_dataset_path)
with open(second_dataset_path, "r") as f:
    mydataset = json.load(f)

# 给mydataset的每一项添加cot_length=1
for item in mydataset:
    item["cot_length"] = 1

    # 添加必要的字段以保持数据格式一致
    if "instruction" not in item:
        item["instruction"] = "Please reason step by step, and put your final answer within \\boxed{}."
    if "input" not in item:
        item["input"] = ""

# 合并两个数据集
merged_dataset = new_dataset + mydataset

# 保存合并后的数据集
output_path = args.output
if not os.path.isabs(output_path):
    output_path = os.path.join(current_dir, output_path)
with open(output_path, "w") as f:
    json.dump(merged_dataset, f, indent=1)

print(f"合并完成! 共有 {len(merged_dataset)} 条数据。")
print(f"数据已保存至: {output_path}")

import os
import json
import re
import argparse
from collections import defaultdict


def process_dataset(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 输出文件路径
    output_path = os.path.join(output_dir, "auto_dataset.json")

    # 收集所有数据点并记录裁剪率
    all_data = []

    # 遍历所有裁剪率文件夹
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 解析裁剪率（过滤非数值文件夹）
        try:
            prune_rate = float(folder)
        except ValueError:
            continue

        # 读取预测文件
        pred_file = os.path.join(folder_path, "samples/predictions.jsonl")
        if not os.path.exists(pred_file):
            continue

        with open(pred_file, "r") as f:
            for line in f:
                data = json.loads(line)
                data["prune_rate"] = prune_rate  # 添加裁剪率字段
                all_data.append(data)

    # 按数据点ID分组
    id_group = defaultdict(list)
    for dp in all_data:
        id_group[dp["id"]].append(dp)

    # 处理每个数据点
    processed = []
    for data_id, group in id_group.items():
        # 筛选正确预测的记录
        correct_records = [dp for dp in group if dp["accuracy"]]
        if not correct_records:
            continue

        # 找到最低有效裁剪率
        min_prune = min(correct_records, key=lambda x: x["prune_rate"])
        print(min_prune["prune_rate"])

        # 将数据转换为新格式
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."

        # 修改prompt中的裁剪率为auto
        input_text = re.sub(r"(<\|eot_id\|>)\d+\.?\d*(<\|eot_id\|>)", r"\1auto\2", min_prune["prompt"])

        # 删除input中的前缀部分
        input_text = re.sub(
            r"<\|im_start\|>system\nYou are a helpful assistant\.<\|im_end\|>\n<\|im_start\|>user\nPlease reason step by step, and put your final answer within \\boxed\{\}\.\n",
            "",
            input_text,
        )
        input_text = re.sub(
            r"<\|im_end\|>\n<\|im_start\|>assistant\n",
            "",
            input_text,
        )

        # 输出部分直接使用原始completion
        output = min_prune.get("model_output", "")

        # 构建新数据点
        new_data = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }

        processed.append(new_data)

    # 写入新数据集 - 使用JSON格式
    with open(output_path, "w") as f:
        json.dump(processed, f, indent=1)  # 使用indent参数美化输出格式

    print(f"处理完成，已生成数据集：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理数据集并生成新格式")
    parser.add_argument("--input-dir", type=str, required=True, help="输入目录路径，包含各裁剪率的子文件夹")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录路径，数据集将保存在此目录下")

    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir)

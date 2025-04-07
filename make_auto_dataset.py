import os
import json
import re
import argparse
import random
from collections import defaultdict


def process_dataset(input_dir, output_dir, balance=True, balance_weight=1.0):
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
    # 统计每种裁剪率的数据点数量
    prune_rate_counts = defaultdict(int)
    
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
            "prune_rate": min_prune["prune_rate"]  # 添加裁剪率字段
        }

        processed.append(new_data)
        
        # 更新裁剪率统计
        prune_rate_counts[min_prune["prune_rate"]] += 1

    # 数据均衡处理
    if balance:
        processed = balance_dataset(processed, prune_rate_counts, balance_weight)

    # 写入新数据集 - 使用JSON格式
    with open(output_path, "w") as f:
        json.dump(processed, f, indent=1)  # 使用indent参数美化输出格式

    print(f"处理完成，已生成数据集：{output_path}")
    
    # 统计均衡后的各裁剪率数据点数量
    balanced_counts = defaultdict(int)
    for item in processed:
        balanced_counts[item["prune_rate"]] += 1
    
    # 打印各裁剪率的数据点数量统计
    print("\n各裁剪率数据点数量统计:")
    for rate, count in sorted(balanced_counts.items()):
        print(f"裁剪率 {rate:.2f}: {count}个数据点")
    print(f"总计: {sum(balanced_counts.values())}个数据点")


def balance_dataset(data, prune_rate_counts, balance_weight=1.0):
    """
    均衡不同裁剪率的数据点数量，使用权重参数控制分布
    
    Args:
        data: 数据列表
        prune_rate_counts: 各裁剪率的样本计数
        balance_weight: 权重因子，越大高裁剪率样本越多，越小低裁剪率样本越多
    """
    # 按裁剪率分组
    prune_groups = defaultdict(list)
    for item in data:
        prune_groups[item["prune_rate"]].append(item)
    
    # 根据裁剪率和权重计算每个组的相对权重
    rates = list(prune_groups.keys())
    weights = {rate: rate**balance_weight for rate in rates}
    total_weight = sum(weights.values())
    
    # 计算总样本数量
    total_samples = sum(len(items) for items in prune_groups.values())
    
    # 根据权重计算每个组的目标数量（至少1个样本）
    target_counts = {rate: max(1, int((weights[rate] / total_weight) * total_samples)) 
                    for rate in rates}
    
    print(f"\n均衡前数据分布: {dict(prune_rate_counts)}")
    print(f"权重因子: {balance_weight}")
    print(f"目标均衡分布: {target_counts}")
    
    # 均衡后的数据集
    balanced_data = []
    
    # 对每个裁剪率组进行处理
    for rate, items in prune_groups.items():
        current_count = len(items)
        target_count = target_counts[rate]
        
        if current_count > target_count:
            # 下采样：随机选择目标数量的数据点
            balanced_data.extend(random.sample(items, target_count))
            print(f"裁剪率 {rate:.2f}: 下采样 {current_count} -> {target_count}")
        else:
            # 上采样：复制数据点直到达到目标数量
            balanced_data.extend(items)
            
            # 然后随机抽样添加剩余需要的数据点
            additional_needed = target_count - current_count
            if additional_needed > 0:
                # 随机抽样可重复的数据点
                additional_samples = random.choices(items, k=additional_needed)
                balanced_data.extend(additional_samples)
                print(f"裁剪率 {rate:.2f}: 上采样 {current_count} -> {target_count}")
    
    return balanced_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理数据集并生成新格式")
    parser.add_argument("--input-dir", type=str, required=True, help="输入目录路径，包含各裁剪率的子文件夹")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录路径，数据集将保存在此目录下")
    parser.add_argument("--balance", action="store_true", default=True, help="是否均衡不同裁剪率的数据量")
    parser.add_argument("--balance-weight", type=float, default=1.0, 
                       help="控制裁剪率与样本数量的关系权重，>1时高裁剪率占比增加，<1时低裁剪率占比增加")

    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir, args.balance, args.balance_weight)
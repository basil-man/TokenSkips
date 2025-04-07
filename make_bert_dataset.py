import json
import glob
import re

def main():
    # 搜索预测结果文件（假设目录结构保持不变）
    file_paths = glob.glob("outputs/Qwen2.5-7B-Instruct/gsm8k/7b/TokenSkip/*/samples/predictions.jsonl")
    # 保存每个预测正确的实例的最小 gamma，键为 content
    correct_min_gamma = {}
    
    for file_path in file_paths:
        # 从文件路径中提取 gamma 值，例如：.../TokenSkip/0.9/samples/predictions.jsonl
        match = re.search(r'/TokenSkip/([\d\.]+)/samples/predictions\.jsonl$', file_path)
        if not match:
            continue
        gamma = float(match.group(1))
        if gamma < 0.5:
            continue
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    continue
                # 对于预测正确的数据（accuracy 为 True）
                if data.get("accuracy") is True:
                    # 提取 messages 数组中第一个元素的 content 字段
                    messages = data.get("messages", [])
                    if messages and isinstance(messages, list):
                        content = messages[0].get("content")
                        if content:
                            if content not in correct_min_gamma or gamma < correct_min_gamma[content]:
                                correct_min_gamma[content] = gamma
    
    # 将结果输出为一个 JSONL 文件，每行一个 JSON 对象，键为 "content" 和 "gamma"
    output_path = "outputs/correct_min_gamma.jsonl"
    with open(output_path, 'w') as out_f:
        for content, gamma in correct_min_gamma.items():
            json_line = json.dumps({"content": content, "gamma": gamma})
            out_f.write(json_line + "\n")
    print(f"结果已保存至 {output_path}")

if __name__ == "__main__":
    main()
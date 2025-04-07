import json

file_path = "outputs/mydataset_compressed_longbench_llmlingua_qwen_7B.json"  # 替换为你的 JSON 文件路径

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 加载整个 JSON 文件为列表或字典
    first_item = data[0]  # 读取第一项（适用于列表）
    print(len(data))
    print(first_item['instruction'])
    print(first_item['input'][:100])

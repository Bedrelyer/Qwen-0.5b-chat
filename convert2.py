import json

# 读取JSON文件
with open('D://xiaosun//record_alpaca.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为训练格式
formatted_data = []
for item in data:
    formatted_data.append({
        "instruction": item["instruction"],
        "input": item["input"],
        "output": item["output"]
    })

# 保存为新的JSON文件
with open('formatted_data.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

import re
import json
from pathlib import Path
from datasets import Dataset
# 加载数据
with open('D://xiaosun//formatted_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为 Hugging Face Dataset 格式
dataset = Dataset.from_dict({
    "instruction": [item["instruction"] for item in data],
    "input": [item["input"] for item in data],
    "output": [item["output"] for item in data]
})
import json
import numpy as np

# 1. 读取原始 JSON 数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 2. Bootstrap 重采样
def bootstrap_resample_json(data, n_samples=None):
    if n_samples is None:
        n_samples = len(data)
    
    indices = np.random.randint(0, len(data), n_samples)
    
    resampled_data = [data[i] for i in indices]
    
    return resampled_data

# 3. 保存新的 JSON 文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 主流程
if __name__ == "__main__":
    # 原始文件路径
    input_file = 'original_dataset.json'
    
    # 读取数据
    data = load_json(input_file)
    print(f"原始数据大小: {len(data)} 条记录")
    
    # Bootstrap 生成新数据集
    new_data = bootstrap_resample_json(data, n_samples=len(data))  # 也可以调整 n_samples
    
    # 新文件路径
    output_file = 'bootstrap_dataset.json'
    
    # 保存
    save_json(new_data, output_file)
    
    print(f"新训练集生成完成！共 {len(new_data)} 条记录，保存到 {output_file}")

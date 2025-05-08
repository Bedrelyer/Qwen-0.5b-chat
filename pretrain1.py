from huggingface_hub import snapshot_download

# 下载整个模型的所有文件
# proxy_on
local_dir = snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", local_dir="./Qwen2.5-0.5B-Instruct")

print(f"Model directory downloaded to: {local_dir}")
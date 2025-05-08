from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

# 模型名称
model_name = "Qwen/Qwen-7B"

# 指定缓存目录
cache_dir = "D:/huggingface_models"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)


# 正确格式示例
new_tokens = [f"<|unused{i}|>" for i in range(100)]  # 必须符合 <|tag|> 格式

unused_mapping = {f"<unused{i}>": f"<|extra_{i}|>" for i in range(100)}
print(tokenizer.special_tokens_map)  # 应包含新增的 additional_special_tokens
print(tokenizer.convert_tokens_to_ids("<|unused0|>"))  # 输出新增的 token_id（如151643 + 原词表大小）
text = "测试<|unused0|>"
encoded = tokenizer(text, return_tensors="pt")
print(encoded["input_ids"])  # 应包含新增的 token_id

# 验证添加后的词汇表
print("<unused0> 的 ID:", tokenizer.convert_tokens_to_ids("<unused0>"))  # 例如输出 151643（原词汇表末尾+1）


vocab = tokenizer.get_vocab()

print("<|extra_0|>" in vocab)  # 输出 True/False
print(tokenizer.convert_tokens_to_ids("▁暴雨"))  # 返回子词ID
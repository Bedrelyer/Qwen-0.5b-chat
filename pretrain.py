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

# 添加新标记









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
special_tokens = tokenizer.all_special_tokens
special_token_ids = tokenizer.all_special_ids
print("<|extra_0|>" in special_tokens)

for token, token_id in zip(special_tokens, special_token_ids):
    vocab[token] = token_id


# 修复 pad_token 设置逻辑
if tokenizer.pad_token is None:
    # 优先使用预定义的扩展标记（如 <|extra_0|>）
    if "<|extra_0|>" in vocab:
        tokenizer.pad_token = "<|extra_0|>"
    # 若预定义扩展标记不可用，回退到 eos_token（需谨慎）
    elif tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        pass
        #raise ValueError("无法设置 pad_token，请手动添加符合格式的特殊标记")

print("Tokenizer 和模型加载成功！")
print("Pad token:", tokenizer.pad_token)
print("Pad token ID:", tokenizer.pad_token_id)













# 加载数据集
dataset = load_dataset('json', data_files='D://xiaosun//formatted_data.json')

# 数据预处理函数
def preprocess_function(examples):
    # 将 instruction、input 和 output 拼接为模型输入
    inputs = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        if input_text:  # 如果 input 不为空
            inputs.append(f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}")
        else:  # 如果 input 为空
            inputs.append(f"Instruction: {instruction}\nOutput: {output}")
    
    # Tokenization（禁用填充）
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=False)
    
    # 将 output 部分作为标签
    labels = tokenizer(examples["output"], max_length=512, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# 对数据集进行预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 动态填充批次数据
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 对于因果语言模型
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,  # 适合指令微调的学习率
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True,  # 启用混合精度训练
    logging_dir="./logs",
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],  # 如果没有验证集，可以用训练集代替
    data_collator=data_collator,  # 使用动态填充
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model("./fine_tuned_qwen")
tokenizer.save_pretrained("./fine_tuned_qwen")
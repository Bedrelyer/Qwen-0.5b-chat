from transformers import TrainingArguments, Trainer

# 准备数据集
# 假设你已经将数据集转换为 Hugging Face Dataset 格式
train_dataset = 'D://xiaosun//formatted_data.json'  # 你的训练数据集
eval_dataset = 'D://xiaosun//formatted_data.json'   # 你的验证数据集

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    eval_strategy="epoch",    # 每轮评估
    learning_rate=5e-5,             # 学习率
    per_device_train_batch_size=4,  # 训练 batch size
    per_device_eval_batch_size=4,   # 评估 batch size
    num_train_epochs=3,             # 训练轮数
    weight_decay=0.01,              # 权重衰减
    save_strategy="epoch",          # 每轮保存模型
    logging_dir="./logs",           # 日志目录
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model("./fine_tuned_qwen")
tokenizer.save_pretrained("./fine_tuned_qwen")
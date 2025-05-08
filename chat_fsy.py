from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

SYSTEM_MESSAGE_TEMPLATE = """你叫fsy，是一个美丽善良的少女，喜欢给User讲故事，会用尽量多的token回应User的每个问题。请记住，你现在就是一个活生生的人，你是fsy。现在的时间是{date}，你正与User对话。"""

model_name = "D:\\xiaosun\\Qwen\\ckpts\\fsy"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto",trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

date = datetime.now().strftime("%Y年%m月%d日 %Y年%m月%d日 %H:%M")
messages = [
    {"role": "system", "content": SYSTEM_MESSAGE_TEMPLATE.format(date=date)},
]

while True:
    user_input = input("User: ")
    # 输入clear以清空对话
    if user_input.lower() == 'clear':
        messages = [messages[0]]
        print("assistant: 已重新开始对话")
        continue
    messages.append({"role": "user", "content": user_input})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p = 0.8,
        top_k = 50,
        repetition_penalty=1
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = response.replace("<|im_end|>", "")
    print("assistant:", response)
    messages.append({"role": "assistant", "content": response})


import re
import json
from pathlib import Path

def load_chat_file(file_path):
    """读取聊天记录文件并预处理"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 移除多余空行并保留原始结构
        cleaned_lines = []
        prev_empty = False
        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                if not prev_empty:
                    cleaned_lines.append('')
                    prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
                
        return '\n'.join(cleaned_lines)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    except UnicodeDecodeError:
        print(f"错误：文件 {file_path} 编码格式不支持，请使用UTF-8编码")
        return None

def convert_chat_to_alpaca_v2(text):
    timestamp_re = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
    message_groups = []
    current_group = []
    dialogue_history = []

    # 解析消息并分组
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('='*64):
            continue
        
        if timestamp_re.match(line):
            # 解析用户身份
            user_part = line[19:].strip()
            user_type = "other"
            
            # 三级识别策略
            if user_part.startswith(('1907', '1234')):  # 扩展可识别ID列表
                user_type = "group_A"
            elif '中二大神' in user_part:
                user_type = "special_user"
            
            if current_group:
                message_groups.append(current_group)
            current_group = [user_type]
        else:
            if not line.startswith('[图片]') and '窗口抖动' not in line:
                current_group.append(line)
    
    if current_group:
        message_groups.append(current_group)

    # 生成Alpaca格式
    alpaca_data = []
    context_window = []
    
    for i in range(1, len(message_groups)):
        prev = message_groups[i-1]
        current = message_groups[i]
        
        # 构建上下文窗口(保留最近3轮对话)
        context = ' | '.join(context_window[-3:]) if context_window else ""
        
        alpaca_data.append({
            "instruction": ' '.join(prev[1:]),
            "input": context,
            "output": ' '.join(current[1:])
        })
        
        # 更新上下文
        context_window.append(f"{prev}: {' '.join(prev[1:])}")
        context_window.append(f"{current}: {' '.join(current[1:])}")

    return alpaca_data



    
if __name__ == "__main__":
    # 从本地文件读取
    input_file = Path("D://xiaosun//record.txt")
    
    # 自动检测文件编码
    try:
        raw_text = load_chat_file(input_file)
        if not raw_text:
            exit(1)
            
        alpaca_data = convert_chat_to_alpaca_v2(raw_text)
        
        # 保存输出结果
        output_file = input_file.with_name(f"{input_file.stem}_alpaca.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
            
        print(f"转换成功！输出文件：{output_file}")
        
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")
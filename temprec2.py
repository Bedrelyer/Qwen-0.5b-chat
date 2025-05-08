import re
import json
from datetime import datetime

def process_chat_to_json(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Initialize variables
    result = []
    current_conversation_id = None
    messages = []
    previous_timestamp = None
    
    # Updated regular expression to match timestamp with possible single-digit hour
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}) (.+)')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Try to match timestamp pattern
        match = timestamp_pattern.match(line)
        if match:
            timestamp = match.group(1)
            sender = match.group(2)
            
            # Normalize timestamp format (ensure hour has two digits)
            timestamp_parts = timestamp.split(' ')
            time_parts = timestamp_parts[1].split(':')
            if len(time_parts[0]) == 1:
                time_parts[0] = '0' + time_parts[0]
            normalized_timestamp = timestamp_parts[0] + ' ' + ':'.join(time_parts)
            
            # Check if we need to start a new conversation (4-hour gap)
            if previous_timestamp:
                previous_dt = datetime.strptime(previous_timestamp, "%Y-%m-%d %H:%M:%S")
                current_dt = datetime.strptime(normalized_timestamp, "%Y-%m-%d %H:%M:%S")
                time_diff = (current_dt - previous_dt).total_seconds() / 3600  # hours
                
                if time_diff >= 4:
                    # Start a new conversation
                    if messages:
                        # Process previous conversation
                        process_conversation(result, current_conversation_id, messages)
                        messages = []
                    current_conversation_id = normalized_timestamp
            
            # If no conversation has been started yet, start one
            if current_conversation_id is None:
                current_conversation_id = normalized_timestamp
            
            previous_timestamp = normalized_timestamp
            
            # Extract the message content
            message_content = []
            j = i + 1
            while j < len(lines) and not timestamp_pattern.match(lines[j].strip()):
                content = lines[j].strip()
                if content and not re.match(r'^\[图片\]$|\[表情\]$|您发送了一个窗口抖动.*$', content):
                    message_content.append(content)
                j += 1
            
            # Skip system messages
            if not sender.startswith("系统消息"):
                role = "assistant" if sender == "PTy0" else "user"
                message_text = "\n".join(message_content)
                # Remove [图片] and [表情] from the message
                message_text = re.sub(r'\[图片\]|\[表情\]', '', message_text)
                
                if message_text.strip():  # Only add non-empty messages
                    messages.append((role, message_text))
            
            i = j
        else:
            i += 1
    
    # Process the last conversation
    if messages and current_conversation_id:
        process_conversation(result, current_conversation_id, messages)
    
    return result

def process_conversation(result, conversation_id, messages):
    # Group consecutive messages from the same sender
    conversation = {
        "id": conversation_id,
        "conversations": []
    }
    
    current_role = None
    current_message = []
    
    for role, message in messages:
        if current_role is None:
            current_role = role
            current_message.append(message)
        elif current_role == role:
            # Same sender, append message
            current_message.append(message)
        else:
            # Different sender, add the grouped message and start a new one
            if current_message:  # Only add non-empty messages
                conversation["conversations"].append({
                    "from": current_role,
                    "value": "\n".join(current_message)
                })
            current_role = role
            current_message = [message]
    
    # Add the last message group
    if current_message:
        conversation["conversations"].append({
            "from": current_role,
            "value": "\n".join(current_message)
        })
    
    # Only add conversations with actual messages
    if conversation["conversations"]:
        result.append(conversation)

def main():
    input_file = "merged_chat_records.txt"  # Change this to your input file path
    output_file = "chat_output.json"  # Change this to your desired output file path
    
    result = process_chat_to_json(input_file)
    
    # Write the result to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion completed. Output saved to {output_file}")

if __name__ == "__main__":
    main()
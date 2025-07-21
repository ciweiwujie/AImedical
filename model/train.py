import torch
from torch.utils.data import DataLoader
import os
import json
import random
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
import datetime
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建结果目录
os.makedirs('result', exist_ok=True)

from params import Params
from medical_tokenizer import MedicalTokenizer
from dataset import MedicalQADataset

# 参数配置类
params = Params()

# 初始化tokenizer
tokenizer = MedicalTokenizer(params.vocab_path)

# 解析医疗数据文件
def parse_medical_data(file_path):
    qa_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳过空行
        if not line:
            i += 1
            continue
            
        # 问题行
        question = line
        i += 1
        
        # 如果还有行，取出答案
        if i < len(lines):
            answer_line = lines[i].strip()
            # 如果下一行非空，则作为答案
            if answer_line:
                answer = answer_line
                qa_pairs.append((question, answer))
                i += 1  # 移动到下一行
            else:
                # 空答案
                qa_pairs.append((question, ""))
        else:
            # 没有答案行
            qa_pairs.append((question, ""))
        
        # 跳过空行（如果有）
        if i < len(lines) and not lines[i].strip():
            i += 1
    
    return qa_pairs

# 数据填充函数
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 填充序列
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100  # 忽略填充位置的损失
    )
    
    # 创建注意力掩码
    attention_mask = (input_ids != tokenizer.pad_token_id).float()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# 计算准确率的函数
def calculate_accuracy(model, data_loader):
    model.eval()
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(params.device)
            masks = batch['attention_mask'].to(params.device)
            labels = batch['labels'].to(params.device)
            
            outputs = model(input_ids=inputs, attention_mask=masks)
            logits = outputs.logits
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            mask = (labels != -100)  # 忽略填充位置
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    return total_correct / total_tokens if total_tokens > 0 else 0

# 绘制训练曲线的函数
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('result/loss_curves.png')
    plt.close()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('result/accuracy_curves.png')
    plt.close()

# 训练函数
def train_model(model, train_loader, valid_loader):
    optimizer = AdamW(model.parameters(), lr=params.learning_rate)
    total_steps = len(train_loader) * params.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    # 记录训练过程中的指标
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(params.epochs):
        print(f"Epoch {epoch+1}/{params.epochs}")
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs = batch['input_ids'].to(params.device)
            masks = batch['attention_mask'].to(params.device)
            labels = batch['labels'].to(params.device)
            
            # 前向传播
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 计算训练集准确率
        train_acc = calculate_accuracy(model, train_loader)
        train_accs.append(train_acc)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                inputs = batch['input_ids'].to(params.device)
                masks = batch['attention_mask'].to(params.device)
                labels = batch['labels'].to(params.device)
                
                outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        
        # 计算验证集准确率
        val_acc = calculate_accuracy(model, valid_loader)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), params.save_path)
            print(f"Saved best model with loss: {best_val_loss:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

# 加载数据
qa_pairs = parse_medical_data(params.train_path)
print(f"Loaded {len(qa_pairs)} QA pairs")

# 划分训练集和验证集 (80%训练, 20%验证)
random.shuffle(qa_pairs)
split_idx = int(len(qa_pairs) * 0.8)
train_pairs = qa_pairs[:split_idx]
valid_pairs = qa_pairs[split_idx:]

train_dataset = MedicalQADataset(train_pairs, tokenizer)
valid_dataset = MedicalQADataset(valid_pairs, tokenizer)

train_loader = DataLoader(
    train_dataset, 
    batch_size=params.batch_size, 
    shuffle=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=params.batch_size,
    collate_fn=collate_fn
)

# 初始化模型
config = GPT2Config.from_json_file(params.model_config)
config.vocab_size = tokenizer.vocab_size
config.loss_type = "ForCausalLMLoss"
model = GPT2LMHeadModel(config)
model.to(params.device)

# 生成响应函数（保持不变）
def generate_response(model, question, max_length=150):
    model.eval()
    
    input_ids = tokenizer.encode(question, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(params.device)
    
    generated = []
    past_key_values = None
    generated_phrases = set()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            
            next_token_logits = modify_logits(
                next_token_logits,
                generated=generated,
                temperature=0.7,
                repetition_penalty=1.5
            )
            
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            if next_token.item() == tokenizer.sep_token_id:
                break
                
            generated.append(next_token.item())
            input_ids = next_token
            past_key_values = outputs.past_key_values
    
    response = tokenizer.decode(generated)
    return postprocess_response(response)

def modify_logits(logits, generated, temperature=0.7, repetition_penalty=1.5):
    logits = logits / temperature
    
    for token_id in set(generated[-10:]):
        logits[:, token_id] /= repetition_penalty
    
    return logits

def postprocess_response(text):
    phrases = []
    current_phrase = []
    
    for char in text:
        if char in ['，', '。', '；']:
            if current_phrase:
                phrases.append(''.join(current_phrase).strip())
                current_phrase = []
        else:
            current_phrase.append(char)
    
    if current_phrase:
        phrases.append(''.join(current_phrase).strip())
    
    seen = set()
    unique_phrases = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique_phrases.append(p)
    
    return '，'.join(unique_phrases[:3])

# 主函数
def main():
    # 1. 训练模型
    print("Starting training at", datetime.datetime.now())
    train_model(model, train_loader, valid_loader)
    print("Training completed at", datetime.datetime.now())
    
    # 2. 加载最佳模型进行推理
    model.load_state_dict(torch.load(params.save_path))
    model.to(params.device)
    
    # 3. 测试几个示例问题
    test_questions = [
        "小孩发烧又吐怎么办？",
        "高甘油三酯血症的就诊科室是什么？",
        "脐周淋巴结肿大怎么处理？"
    ]
    
    print("\nTesting model:")
    for q in test_questions:
        response = generate_response(model, q)
        print(f"问题: {q}")
        print(f"回答: {response}\n")
        
if __name__ == "__main__":
    main()
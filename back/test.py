import os
import torch
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Config
import datetime
import sys
import random
# 在导入部分添加
from flask_cors import CORS

# 添加路径以便导入model模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from medical_tokenizer import MedicalTokenizer

app = Flask(__name__)
CORS(app)

# 测试配置
class TestParams:
    def __init__(self):
        # 修正文件路径
        self.vocab_path = '../model/txt/vocab.txt'
        self.model_config = '../model/json/config.json'
        self.model_path = '../model/bin/medical_model.bin'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = 256

params = TestParams()

# 生成响应函数
def generate_response(model, question, max_length=150):
    model.eval()
    
    # 初始化tokenizer
    tokenizer = MedicalTokenizer(params.vocab_path)
    
    input_ids = tokenizer.encode(question, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(params.device)
    
    generated = []
    past_key_values = None
    
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

# 加载tokenizer和模型
def load_model_and_tokenizer(params):
    # 初始化tokenizer
    tokenizer = MedicalTokenizer(params.vocab_path)
    
    # 初始化模型
    config = GPT2Config.from_json_file(params.model_config)
    config.vocab_size = tokenizer.vocab_size
    model = GPT2LMHeadModel(config)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(params.model_path, map_location=params.device))
    model.to(params.device)
    model.eval()
    
    return model, tokenizer

# 交互式问答函数
def interactive_qa(model, tokenizer):
    print("\n医疗问答系统 (输入'退出'或'quit'结束)")
    print("=" * 50)
    print("您可以输入任何医疗相关问题，例如：")
    print("- 小孩发烧怎么办？")
    print("- 高血压吃什么药？")
    print("- 糖尿病有什么症状？")
    print("=" * 50)
    
    while True:
        try:
            # 获取用户输入
            question = input("\n请输入您的医疗问题: ").strip()
            
            # 退出条件
            if question.lower() in ['退出', 'exit', 'quit']:
                print("\n感谢使用医疗问答系统，再见！")
                break
                
            # 空输入处理
            if not question:
                print("问题不能为空，请重新输入")
                continue
                
            # 检查问题是否以问号结尾
            if not question.endswith(('？', '?')):
                question += '？'
                
            # 生成回答
            print("\n正在生成回答，请稍候...")
            response = generate_response(model, question)
            
            # 显示结果
            print("\n问题:", question)
            print("回答:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # 提供继续选项
            cont = input("\n是否继续提问？(y/n): ").strip().lower()
            if cont not in ['y', 'yes', '是']:
                print("\n感谢使用医疗问答系统，再见！")
                break
                
        except KeyboardInterrupt:
            print("\n检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            print("请重新尝试或联系管理员")

@app.route('/predict', methods=['POST'])
def predict():
    data=request.json or {}
    question=data.get('question','').strip()
    print(f"收到问题: {question}")
    response=generate_response(model,question)
    return jsonify({"answer":response})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "医疗问答系统运行正常"})

if __name__ == "__main__":
    # 加载模型和tokenizer
    print("正在加载医疗问答模型...")
    model, tokenizer = load_model_and_tokenizer(params)
    print("模型加载完成！")

    # 启动 Flask 服务
    print("启动医疗问答 API 服务...")
    print("服务地址: http://localhost:5000")
    print("API端点: /predict (POST)")
    print("健康检查: /health (GET)")
    app.run(host='0.0.0.0', port=5000, debug=False)
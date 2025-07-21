import torch
from transformers import GPT2LMHeadModel, GPT2Config
from medical_tokenizer import MedicalTokenizer
from train import generate_response  

# 测试配置
class TestParams:
    def __init__(self):
        self.vocab_path = './txt/vocab.txt'
        self.model_config = './json/config.json'
        self.model_path = './bin/medical_model.bin'
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.max_len = 256

params = TestParams()

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

if __name__ == "__main__":
    # 加载模型和tokenizer
    print("正在加载医疗问答模型...")
    model, tokenizer = load_model_and_tokenizer(params)
    print("模型加载完成！")
    
    # 启动交互式问答
    interactive_qa(model, tokenizer)
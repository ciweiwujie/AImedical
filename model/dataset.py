import torch
from torch.utils.data import Dataset
from params import Params

# 创建输入序列
def create_input_sequence(question, answer, max_len, tokenizer):
    """创建问答序列：[CLS]问题[SEP]答案[SEP]"""
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    
    # 添加特殊token
    input_ids = [tokenizer.cls_token_id] + question_ids + [tokenizer.sep_token_id] + answer_ids + [tokenizer.sep_token_id]
    
    # 截断或填充
    if len(input_ids) > max_len:
        # 保留问题部分，截断答案
        keep_len = max_len - len(question_ids) - 2  # -2 for [CLS] and first [SEP]
        if keep_len > 0:
            input_ids = input_ids[:len(question_ids)+2] + answer_ids[:keep_len] + [tokenizer.sep_token_id]
        else:
            # 问题太长，只保留问题部分
            input_ids = input_ids[:max_len-1] + [tokenizer.sep_token_id]
    else:
        # 填充
        padding = [tokenizer.pad_token_id] * (max_len - len(input_ids))
        input_ids += padding
    
    return input_ids


# 自定义数据集类
class MedicalQADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer):
        self.qa_pairs = qa_pairs
        self.params = Params()
        self.tokenizer = tokenizer

        
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        input_ids = create_input_sequence(question, answer, self.params.max_len, self.tokenizer)
        labels = input_ids.copy()
        return torch.tensor(input_ids), torch.tensor(labels)
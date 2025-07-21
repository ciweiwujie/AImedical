class MedicalTokenizer:
    def __init__(self, vocab_path):
        # 初始化映射字典
        self.vocab = {}
        self.ids_to_tokens = {}
        
        # 从文件加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
        
        # 构建词汇表映射
        for idx, token in enumerate(tokens):
            self.vocab[token] = idx
            self.ids_to_tokens[idx] = token
        
        # 设置特殊token属性
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'
        
        self.pad_token_id = self.vocab.get(self.pad_token)
        self.cls_token_id = self.vocab.get(self.cls_token)
        self.sep_token_id = self.vocab.get(self.sep_token)
        self.unk_token_id = self.vocab.get(self.unk_token)
        self.mask_token_id = self.vocab.get(self.mask_token)
        
        self.vocab_size = len(self.vocab)
    
    def tokenize(self, text):
        """字符级分词，保留特殊符号"""
        # 简单按字符分割，但保留连续的特殊符号（如标点）
        tokens = []
        current_token = []
        
        for char in text:
            if char in self.vocab:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                tokens.append(char)
            else:
                current_token.append(char)
        
        if current_token:
            tokens.append(''.join(current_token))
        
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        """将token列表转换为id列表"""
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """将id列表转换为token列表"""
        return [self.ids_to_tokens.get(id, self.unk_token) for id in ids]
    
    def encode(self, text, add_special_tokens=True, max_length=None):
        """编码文本为ID序列"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # 处理长度限制
        if max_length:
            if len(input_ids) > max_length:
                # 保留开头和SEP
                keep_len = max_length - 1
                input_ids = input_ids[:1] + input_ids[-keep_len:] if keep_len > 0 else input_ids[:max_length]
            else:
                # 填充
                padding = [self.pad_token_id] * (max_length - len(input_ids))
                input_ids += padding
        
        return input_ids
    
    def decode(self, ids):
        """解码ID序列为文本"""
        tokens = self.convert_ids_to_tokens(ids)
        # 过滤特殊token
        tokens = [t for t in tokens if t not in [
            self.cls_token, 
            self.sep_token, 
            self.pad_token
        ] and not t.startswith('[unused')]
        return ''.join(tokens)
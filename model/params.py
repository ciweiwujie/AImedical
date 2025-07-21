import torch

class Params:
    def __init__(self):
        self.vocab_path = './txt/vocab.txt'
        self.model_config = './json/config.json'
        self.train_path = './txt/medical.txt'
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.max_len = 256
        self.epochs = 15
        self.learning_rate = 5e-5
        self.save_path = './bin/medical_model.bin'
        self.pretrained_model = None  # 从头开始训练
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, OpenAIGPTModel, OpenAIGPTTokenizer

class BERT(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BERT, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)
        self.softmax = nn.Softmax()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = torch.mean(outputs["last_hidden_state"], dim=1)
        x = self.linear(cls_output)
        x = self.softmax(x)
        return x
    
class GPT(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GPT, self).__init__()
        self.model = OpenAIGPTModel.from_pretrained("openai-community/openai-gpt")
        self.tokenizer = BertTokenizer.from_pretrained("openai-community/openai-gpt")
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)
        self.softmax = nn.Softmax()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = torch.mean(outputs["last_hidden_state"], dim=1)
        x = self.linear(cls_output)
        x = self.softmax(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.emb = nn.Embedding(num_embeddings=len(self.tokenizer), embedding_dim=input_size)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(self.emb(input_ids))[0]
        x = torch.mean(outputs, dim=1)
        x = self.linear(x)
        x = self.softmax(x)
        return x
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.emb = nn.Embedding(num_embeddings=len(self.tokenizer), embedding_dim=input_size)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(self.emb(input_ids))[0]
        x = torch.mean(outputs, dim=1)
        x = self.linear(x)
        x = self.softmax(x)
        return x
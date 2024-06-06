import torch
from torch.utils.data import  Dataset
        
class TextDataSet(Dataset):
    def __init__(self, text, label, tokenizer, max_length):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        encoding = self.tokenizer.encode_plus(
            self.text[index],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label[index], dtype=torch.long)
        }
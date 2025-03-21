import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
import os
from transformers import BertForSequenceClassification, BertTokenizer
import json
from collections import Counter
import numpy as np


PTH_PATH="/data/haominpeng/Work/dian/exam/Bert/result/best_model_fl_0.34.pth"
DEVICE = torch.device('cuda')
torch.cuda.set_device(0)

class scoreDataset(Dataset):
    def __init__(self, texts, tokenizer, label, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.label = label
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label[idx]
        #encoding = self.tokenizer(text, max_length=256,  return_tensors='pt')
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len, # 明确指定长度
            padding='max_length',    # 填充到最大长度
            truncation=True,         # 启用截断
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long), # 转换为Tensor
            'idx': idx 
        }

def load_jsonl_data(file_path):
    text=[]
    label=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text.append(data['text'])
            label.append(data['point']-1)
    return text,label


PTH_PATH = "/data/haominpeng/Work/dian/exam/Bert/result/best_model_0.36_0.76_277_fl.pth"
model_path = "/data/haominpeng/Work/dian/exam/Bert/google-bert/bert-base-chinese"

model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=10,  
    hidden_dropout_prob=0.5,  
    attention_probs_dropout_prob=0.5,
    classifier_dropout=0.5
)
model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))

model.to(DEVICE)
tokenizer = BertTokenizer.from_pretrained(model_path)

test_texts,test_labels = load_jsonl_data('/data/haominpeng/Work/dian/exam/Bert/catch/test.jsonl')

# 统计文本长度分布
text_lengths = [len(tokenizer.tokenize(text)) for text in test_texts]
print(f"""
平均长度: {np.mean(text_lengths):.1f}
最短长度: {np.min(text_lengths)}
最长长度: {np.max(text_lengths)}
95%分位数: {np.percentile(text_lengths, 95):.1f}
""")

# 推荐取值策略
max_length = 256

test_dataset = scoreDataset(test_texts, tokenizer, test_labels, max_length)

batch_size = 32

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

learning_rate = 5e-5


# 修改梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
best_accuracy = 0


model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    correct_3 = 0
    all_indices = []
    all_labels = []
    all_preds = []
    for batch in test_loader:
        indices = batch['idx']
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
        correct_3 += (predicted == labels).sum().item()
        correct_3 += (predicted == (labels+1)).sum().item()
        correct_3 += (predicted == (labels-1)).sum().item()
        val_accuracy=correct/total

        all_indices.extend(indices.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
    
    for idx, label, pred in zip(all_indices, all_labels, all_preds):
        original_text = test_texts[idx]

        print(f"句子: {original_text}")
        print(f"真实标签: {label+1}, 预测标签: {pred+1}\n")
    print(f'分类：Loss: {loss.item():.4f} Accuracy: {val_accuracy:.4f} Accuracy_near3: {correct_3/total:.4f}')

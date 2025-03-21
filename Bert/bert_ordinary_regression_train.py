import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
import os
from transformers import BertForSequenceClassification, BertTokenizer
import json
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

def load_jsonl_data(file_path):
    text=[]
    label=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text.append(data['text'])
            label.append(data['point']-1)
    return text,label

test_texts,test_labels = load_jsonl_data('/data/haominpeng/Work/dian/exam/Bert/catch/test.jsonl')

MODEL_PATH = "/data/haominpeng/Work/dian/exam/Bert/result/acc_0.42_llm3_model.pth"
BERT_PATH = "/data/haominpeng/Work/dian/exam/Bert/google-bert/bert-base-chinese"

DEVICE = torch.device('cuda')
torch.cuda.set_device(2)

def logits_to_class(logits):
    """将序数回归的9维输出转换为0-9的类别（修正版）"""
    binary_preds = (torch.sigmoid(logits) >= 0.5).long()
    return torch.clamp(binary_preds.sum(dim=1), 0, 9)

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
            max_length=self.max_len,          # 明确指定长度
            padding='max_length',    # 填充到最大长度
            truncation=True,         # 启用截断
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # 转换为Tensor
        }


class BertForOrdinalRegression(BertForSequenceClassification):
    def __init__(self, config, loss_fn=None):  # 新增loss_fn参数
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels - 1)
        self.loss_fn = loss_fn 

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, 9]
            
        return {'logits': logits}


tokenizer = BertTokenizer.from_pretrained(BERT_PATH)


text_lengths = [len(tokenizer.tokenize(text)) for text in test_texts]
print(f"""
平均长度: {np.mean(text_lengths):.1f}
最短长度: {np.min(text_lengths)}
最长长度: {np.max(text_lengths)}
95%分位数: {np.percentile(text_lengths, 95):.1f}
""")

# # 推荐取值策略
# max_length = min(
#     int(np.percentile(text_lengths, 95) + 10),  # 覆盖95%样本 + 缓冲
#     512  # BERT的最大限制
# )
max_length=190

test_dataset = scoreDataset(test_texts, tokenizer, test_labels, max_length)

# 计算类别权重
# class_counts = np.bincount(test_labels)
# class_weights = 1. / class_counts
#sample_weights = class_weights[test_labels]
#print(f'Class counts: {class_counts},shape: {class_counts.shape}')
#print(f'Class weights: {class_weights},shape: {class_weights.shape}')
#print(f'Sample weights: {sample_weights},  shape: {sample_weights.shape}')

batch_size = 32

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BertForOrdinalRegression.from_pretrained(
    BERT_PATH,
    num_labels=10,  # 实际输出维度为9
    hidden_dropout_prob=0.5,
    attention_probs_dropout_prob=0.5,
    classifier_dropout=0.5,
    #loss_fn=OrdinalRegressionLoss(10)
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)

# 验证阶段
model.eval()
all_labels = []
all_preds = []
#total_loss = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs['logits']
        preds = logits_to_class(logits)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy()) 
        #loss = outputs['loss']  # 直接使用模型计算的损失
        #total_loss += loss.item()

# 计算指标
#avg_loss = total_loss / len(test_loader)
accuracy = accuracy_score(all_labels, all_preds)

near3_acc = sum(1 for t, p in zip(all_labels, all_preds) if abs(t - p) <= 1) / len(all_labels)

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

print("Class-wise Accuracy:")
class_acc = {}
for score in range(10):
    mask = np.array(all_labels) == score
    total = sum(mask)
    acc = np.mean(np.array(all_preds)[mask] == score) if total >0 else 0.0
    class_acc[score + 1] = acc
    print(f"Score {score+1}: {acc:.2%}", end=' ')

print(f'\n有序回归： Acc: {accuracy:.4f} | Near3 Acc: {near3_acc:.4f}\n')

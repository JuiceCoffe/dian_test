import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
import os
from transformers import BertForSequenceClassification, BertTokenizer
import json
from collections import Counter
import numpy as np

DEVICE = torch.device('cuda')
torch.cuda.set_device(3)


class SmoothLoss(nn.Module):
    def __init__(self, num_classes=10, sigma=1.5, a=0.3,b=0.3,c=0.3, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.a = a
        self.b=b
        self.c=c
        self.reduction = reduction


    def forward(self, logits, targets):
        #device = logits.device  # 统一使用logits所在的设备       
        model_probs = F.softmax(logits, dim=1)
        
        # 基础交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 向量化生成高斯目标概率
        distances = torch.arange(self.num_classes, device=DEVICE).view(1, -1) - targets.view(-1, 1)
        probs = torch.exp(-distances.pow(2) / (2 * self.sigma**2)) + 1e-8
        target_probs = probs / probs.sum(dim=1, keepdim=True)
        kl_loss = F.kl_div(target_probs.log(), model_probs,  reduction='none').sum(dim=1)
        
        # 计算序数惩罚项
        outer_probs = model_probs.unsqueeze(2) * model_probs.unsqueeze(1)  # [batch, c, c]
        ordinal_weights = torch.abs(torch.arange(self.num_classes, device=DEVICE).view(1, -1) - torch.arange(self.num_classes, device=DEVICE).view(-1, 1)).float()# 动态生成序数权重矩阵 |i-j|
        ordinal_penalty = (outer_probs * ordinal_weights).sum(dim=(1,2)) / self.num_classes
        
        total_loss = self.a * ce_loss + self.b * kl_loss +  self.c * ordinal_penalty
        return total_loss.mean()

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

def load_jsonl_data(file_path):
    text=[]
    label=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text.append(data['text'])
            label.append(data['point']-1)
    return text,label


model_path = "/data/haominpeng/Work/dian/exam/Bert/google-bert/bert-base-chinese"

model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=10,  
    hidden_dropout_prob=0.5,  
    attention_probs_dropout_prob=0.5,
    classifier_dropout=0.5
)

model.to(DEVICE)
tokenizer = BertTokenizer.from_pretrained(model_path)


train_texts,train_labels = load_jsonl_data('/data/haominpeng/Work/dian/exam/Bert/catch/combined_60049_cleaned_low.jsonl')
test_texts,test_labels = load_jsonl_data('/data/haominpeng/Work/dian/exam/Bert/catch/test.jsonl')

# 统计文本长度分布（示例代码）
text_lengths = [len(tokenizer.tokenize(text)) for text in train_texts]
print(f"""
平均长度: {np.mean(text_lengths):.1f}
最短长度: {np.min(text_lengths)}
最长长度: {np.max(text_lengths)}
95%分位数: {np.percentile(text_lengths, 95):.1f}
""")

# 推荐取值策略
max_length = min(
    int(np.percentile(text_lengths, 95) + 10),  # 覆盖95%样本 + 缓冲
    512  # BERT的最大限制
)

train_dataset = scoreDataset(train_texts, tokenizer, train_labels, max_length)
test_dataset = scoreDataset(test_texts, tokenizer, test_labels, max_length)

# 计算类别权重
class_counts = np.bincount(train_labels)
class_weights = 1. / class_counts
sample_weights = class_weights[train_labels]

print(f'Class counts: {class_counts},shape: {class_counts.shape}')
print(f'Class weights: {class_weights},shape: {class_weights.shape}')
print(f'Sample weights: {sample_weights},  shape: {sample_weights.shape}')
train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

batch_size = 32
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    #sampler=train_sampler,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def print_sampler_distribution(dataloader, class_names=None):
    all_labels = []

    for batch in dataloader:
        labels = batch['labels'].cpu().numpy()
        all_labels.extend(labels.tolist())
    
    total_samples = len(all_labels)
    class_counts = Counter(all_labels)
    
    # 转换为排序后的字典
    sorted_counts = dict(sorted(class_counts.items()))

    print("\n" + "="*40)
    print(f"采样后数据集总大小: {total_samples}")
    print("-"*40)
    print("{:<10} {:<15} {:<15}".format('Class', 'Count', 'Percentage'))
    
    for cls, count in sorted_counts.items():
        percentage = count / total_samples * 100
        class_name = class_names[cls] if class_names else str(cls)
        print("{:<10} {:<15} {:<15.2f}%".format(class_name, count, percentage))
    
    print("="*40 + "\n")

print_sampler_distribution(train_loader)


learning_rate = 4e-5
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)
epochs = 30
total_steps = len(train_loader) * epochs
warmup_steps=0.1*total_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

'''            增大                                减小
sigma	目标分布更平滑，允许更大预测偏差	    目标分布尖锐，要求严格对齐真实标签        调整的是kv散度的高斯化
'''
A = 0.3
B = 0.6
C = 0.1
loss_fn=SmoothLoss(num_classes=10,sigma=0.65,a = A, b = B, c = C)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

best_accuracy = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct_train = 0  
    total_train = 0    
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits=outputs.logits
        loss =loss_fn(logits,labels)
        epoch_loss += loss.item()
        
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    print(f'Epoch {epoch + 1}/{epochs}, avg_Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {correct_train/total_train:.4f}') 

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        correct_3 = 0
        for batch in test_loader:

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss =loss_fn(logits,labels)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_3 += (predicted == labels).sum().item()
            correct_3 += (predicted == (labels+1)).sum().item()
            correct_3 += (predicted == (labels-1)).sum().item()
            val_accuracy=correct/total
            f3_acc=correct_3/total
        print(f'分类：Loss: {loss.item():.4f} Accuracy: {val_accuracy:.4f} Accuracy_near3: {f3_acc:.4f}')
        if ( (val_accuracy>=0.35 and f3_acc>=0.72)or(val_accuracy>=0.3 and f3_acc>=0.8)or(f3_acc>=0.85)):
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f'/data/haominpeng/Work/dian/exam/Bert/result/best_model_{best_accuracy}_{f3_acc}_{max_length}_fl.pth')
            print('\nModel saved')
    print('')

print('A:',A,' B:',B,' C:',C)
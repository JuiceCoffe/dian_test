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


train_texts,train_labels = load_jsonl_data('/data/haominpeng/Work/dian/exam/Bert/catch/combined_45457_cleaned_num.jsonl')
test_texts,test_labels = load_jsonl_data('/data/haominpeng/Work/dian/exam/Bert/catch/test.jsonl')
model_path = "/data/haominpeng/Work/dian/exam/Bert/google-bert/bert-base-chinese"

DEVICE = torch.device('cuda')
torch.cuda.set_device(3)

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
        
        loss = self.loss_fn(logits, labels)
            
        return {'logits': logits, 'loss': loss}


# model = BertForSequenceClassification.from_pretrained(
#     model_path,
#     num_labels=10,  
#     hidden_dropout_prob=0.5,  
#     attention_probs_dropout_prob=0.5,
#     classifier_dropout=0.5
# )

tokenizer = BertTokenizer.from_pretrained(model_path)

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
    #shuffle=True, 
    sampler=train_sampler,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#序数回归损失    
class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes, weight=None, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('thresholds', torch.arange(1, num_classes, dtype=torch.float32)) 
        
        # 处理类别权重
        if weight is not None:
            self.weight = torch.as_tensor(weight, dtype=torch.float32)
        else:
            self.weight = None
            
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        前向计算
        参数：
            logits: 模型输出 [batch_size, num_thresholds]
            targets: 真实标签 [batch_size]（取值范围0到num_classes-1）
        """
        device = logits.device  # 获取当前设备
        
        # 确保 self.thresholds 在相同设备上
        thresholds = self.thresholds.to(device)

        # 将标签转换为二元比较矩阵 [batch_size, num_thresholds]
        targets = targets.view(-1, 1).to(device)  # 确保 targets 也在相同设备上
        targets_expanded = (targets.view(-1, 1) >= self.thresholds).float()
        
        # 计算每个阈值的二元交叉熵损失
        loss_per_threshold = F.binary_cross_entropy_with_logits(
            logits,
            targets_expanded,
            weight=self.weight.to(device) if self.weight is not None else None,
            reduction='none'
        )
        
        # 计算样本维度的平均损失
        loss_per_sample = loss_per_threshold.mean(dim=1)
        
        # 最终损失计算
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


model = BertForOrdinalRegression.from_pretrained(
    model_path,
    num_labels=10,  # 实际输出维度为9
    hidden_dropout_prob=0.5,
    attention_probs_dropout_prob=0.5,
    classifier_dropout=0.5,
    loss_fn=OrdinalRegressionLoss(10)
).to(DEVICE)


learning_rate = 3e-5 
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)



from collections import Counter

def print_sampler_distribution(dataloader, class_names=None):
    """
    打印采样后的类别分布和数据集总大小
    参数：
        dataloader: 使用sampler的DataLoader
        class_names: 可选的类别名称列表
    """
    all_labels = []
    
    # 遍历数据加载器收集标签
    for batch in dataloader:
        labels = batch['labels'].cpu().numpy()
        all_labels.extend(labels.tolist())
    
    # 统计分布
    total_samples = len(all_labels)
    class_counts = Counter(all_labels)
    
    # 转换为排序后的字典
    sorted_counts = dict(sorted(class_counts.items()))
    
    # 打印结果
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

# 使用加权焦点损失
class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return ( (1 - pt) ** self.gamma * ce_loss ).mean()


# 修改梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

epochs = 20
total_steps = len(train_loader) * epochs
warmup_steps=0.1*total_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

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
        logits = outputs['logits']
        loss=outputs['loss']
        epoch_loss += loss.item()

        #_, predicted = torch.max(logits, 1)
        predicted = logits_to_class(logits)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    print(f'Epoch {epoch + 1}/{epochs}, avg_Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {correct_train/total_train:.4f}') 

    # 验证阶段
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0

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
            loss = outputs['loss']  # 直接使用模型计算的损失
            total_loss += loss.item()

    # 计算指标
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    near3_acc = sum(1 for t, p in zip(all_labels, all_preds) if abs(t - p) <= 1) / len(all_labels)

    # 新增混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # 打印类别准确率（原有代码）
    print("Class-wise Accuracy:")
    class_acc = {}
    for score in range(10):
        mask = np.array(all_labels) == score
        total = sum(mask)
        acc = np.mean(np.array(all_preds)[mask] == score) if total >0 else 0.0
        class_acc[score + 1] = acc
        print(f"Score {score+1}: {acc:.2%}", end=' ')

    print(f'\n有序回归：Val Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Near3 Acc: {near3_acc:.4f}\n')
    if(accuracy>=0.35 and near3_acc>=0.70):
        best_accuracy=accuracy
        print("model saved!!!\n")
        torch.save(model.state_dict(), f'/data/haominpeng/Work/dian/exam/Bert/result/acc_{best_accuracy:.2f}_{near3_acc:.2f}_{max_length}_llm3_model.pth')
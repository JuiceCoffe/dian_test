import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
import os
from transformers import BertForSequenceClassification, BertTokenizer  # 保持原模型类
import json
import numpy as np
from collections import Counter

DEVICE = torch.device('cuda')
torch.cuda.set_device(1)

# def print_class_accuracy(preds, labels):
#     # 将 CUDA 张量移动到 CPU 并转换为 numpy 数组
#     labels = torch.round(labels).cpu().numpy()  # 先移动到 CPU
#     preds = torch.round(preds).cpu().numpy()
    
#     num_labels = np.zeros(10)
#     correct_labels = np.zeros(10)
    
#     for i, label in enumerate(labels):
#         label_int = int(label)  # 确保是 Python 原生整数类型
#         num_labels[label_int] += 1
#         if label_int == preds[i]:
#             correct_labels[label_int] += 1
            
#     for i in range(10):
#         if num_labels[i] == 0:
#             print(f'label {i+1}: no samples', end='  ')
#         else:
#             print(f'label {i+1}: {correct_labels[i]/num_labels[i]:.4f}', end='  ')
#     print()  # 添加换行

def regression_metrics(preds, labels):
    mae = torch.mean(torch.abs(preds - labels)).item()
    rmse = torch.sqrt(torch.mean((preds - labels)**2)).item()
    return mae, rmse

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
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,          # <--- 修正变量名 maxlen -> max_len
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)  # <--- 改为float类型
        }

def load_jsonl_data(file_path):
    text=[]
    label=[]
    int_label=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text.append(data['text'])
            label.append(float(data['point'])) 
    return text,label

model_path = "/data/haominpeng/Work/dian/exam/Bert/google-bert/bert-base-chinese"

# 修改模型为回归模式
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1, 
    problem_type="regression" ,
    hidden_dropout_prob=0.5,  
    attention_probs_dropout_prob=0.5,
    classifier_dropout=0.5
)

model.to(DEVICE)
tokenizer = BertTokenizer.from_pretrained(model_path)

train_texts,train_labels = load_jsonl_data('/data/haominpeng/Work/dian/exam/Bert/catch/combined_45457_pernum=1200.jsonl')
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

# # 计算类别权重
# class_counts = np.bincount(train_labels)
# class_weights = 1. / class_counts
# sample_weights = class_weights[train_labels]
# print(f'Class counts: {class_counts},shape: {class_counts.shape}')
# print(f'total num = {len(train_labels)}')
# print(f'Class weights: {class_weights},shape: {class_weights.shape}')
# print(f'Sample weights: {sample_weights},  shape: {sample_weights.shape}')

# train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


train_dataset = scoreDataset(train_texts, tokenizer, train_labels, max_length)
test_dataset = scoreDataset(test_texts, tokenizer, test_labels, max_length)

batch_size = 32
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    # sampler=train_sampler,
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

learning_rate=3e-5
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=learning_rate,
                              weight_decay=0.01)

epochs = 20
total_steps = len(train_loader) * epochs
warmup_steps=0.1*total_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
loss_fn = torch.nn.functional.mse_loss  

# 定义评估指标函数
def rounded_accuracy(preds, labels):
    labels = torch.round(labels)  
    preds = torch.round(preds) 
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def print_distribution(dataloader, class_names=None):
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

print_distribution(train_loader)

# 训练循环修改
best_acc = 0
best_loss=100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_logits = []
    epoch_labels = []

    for i,batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()  # <--- 去掉多余维度 [batch_size, 1] => [batch_size]
        # 手动计算MSE损失
        epoch_logits.append(logits)
        epoch_labels.append(labels)

        loss = loss_fn(logits, labels)  # <--- 替换损失函数
        epoch_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    train_metric = rounded_accuracy(torch.cat(epoch_logits), torch.cat(epoch_labels))
    print(f'Epoch {epoch + 1}/{epochs}, avg_Loss: {epoch_loss/len(train_loader):.4f}, Rounded Accuracy: {train_metric:.4f}')

    # 验证循环
    model.eval()
    with torch.no_grad():
        test_loss = 0
        all_preds = []
        all_labels = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            test_loss += loss_fn(logits, labels)
            
            all_preds.append(logits)
            all_labels.append(labels)
        
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        #print_class_accuracy(preds,labels)
        test_acc = rounded_accuracy(preds, labels)

        diff = torch.abs(preds - labels)
        test_acc3 = (diff <= 1).sum().item() / len(labels)

        mae, rmse = regression_metrics(preds, labels)
        print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        # test_acc3=test_acc
        # test_acc3+=rounded_accuracy(preds+1, labels)
        # test_acc3+=rounded_accuracy(preds-1, labels)
    
    print(f'回归：test:   avg_loss:{test_loss/len(test_loader):.4f} f1_Accuracy: {test_acc:.4f} f3_Accuracy: {test_acc3:.4f}\n')
    if (test_acc > best_acc or test_loss < best_loss):

        if test_acc > best_acc:
            best_acc = test_acc
        if test_loss < best_loss:
            best_loss = test_loss

        if best_acc>=0.34 and test_acc3>=0.72:
            torch.save(model.state_dict(), f'/data/haominpeng/Work/dian/exam/Bert/result/best_model_f1_{test_acc}_f3_{test_acc3}_loss_{best_loss}_len{max_length}_hg.pth')
            print('Model saved!\n')
    
import torch
from torch import nn
from torch.nn import functional as F

num_classes=5
DEVICE=torch.device('cpu')
sigma=0.7

targets=torch.tensor([3])
model_probs=F.softmax(torch.tensor([[0.3,0.3,0.3,0.7,0.3]]),dim=1)

distances = torch.arange(num_classes, device=DEVICE).view(1, -1) - targets.view(-1, 1)
probs = torch.exp(-distances.pow(2) / (2 * sigma**2)) + 1e-8
target_probs = probs / probs.sum(dim=1, keepdim=True)

print('model_probs:\n',model_probs,'\n')

print('distances:\n',distances,'\n')
print('target_probs:\n',target_probs,'\n')


outer_probs = model_probs.unsqueeze(2) * model_probs.unsqueeze(1)  # [batch, c, c]
print('outer_probs:\n',outer_probs,'\n')
ordinal_weights = torch.abs(torch.arange(num_classes, device=DEVICE).view(1, -1) - torch.arange(num_classes, device=DEVICE).view(-1, 1)).float()# 动态生成序数权重矩阵 |i-j|
print('ordinal_weights:\n',ordinal_weights,'\n')
ordinal_penalty = (outer_probs * ordinal_weights).sum(dim=(1,2)) /num_classes
print('outer_probs * ordinal_weights:\n',(outer_probs * ordinal_weights),'\n')
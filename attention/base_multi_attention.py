import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, output_dim):
        super().__init__()
        assert emb_dim%num_heads == 0
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** 0.5
        
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
        
        self.Wo = nn.Linear(emb_dim, output_dim)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        #print('x:',x.shape)
        Q = self.Wq(x) # [B, L, D]
        K = self.Wk(x)
        V = self.Wv(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        attn_weights = self.softmax(attn_scores)
        
        z = torch.matmul(attn_weights, V)  # [B, H, L, D/H]
        
        z=z.transpose(1,2)# [B, L, H, D/H]
        z=z.contiguous()
        z=z.view(batch_size, seq_len, self.emb_dim) # [B, L, D]
        z=z.contiguous()

        output = self.Wo(z)

        return output,K,V
    
class MultiQueryAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, output_dim):
        super().__init__()
        assert emb_dim%num_heads == 0
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** 0.5
        
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, self.head_dim)
        self.Wv = nn.Linear(emb_dim, self.head_dim)
        
        self.Wo = nn.Linear(emb_dim, output_dim)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.Wq(x) # [B, L, D]
        K = self.Wk(x)
        V = self.Wv(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        K = K.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        V = V.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2) # [B, H, 1, D/H]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        attn_weights = self.softmax(attn_scores)
        
        z = torch.matmul(attn_weights, V)  # [B, H, L, D/H]
        
        z=z.transpose(1,2)# [B, L, H, D/H]
        z=z.contiguous()
        z=z.view(batch_size, seq_len, self.emb_dim) # [B, L, D]
        z=z.contiguous()

        output = self.Wo(z)

        return output,K,V
    
class GroupedQueryAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, output_dim, group_size):
        super().__init__()
        assert emb_dim%num_heads == 0
        assert num_heads%group_size == 0 
        
        self.group_size = group_size
        self.group_num = num_heads//group_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** 0.5
        
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, self.group_num*self.head_dim)
        self.Wv = nn.Linear(emb_dim, self.group_num*self.head_dim)
        
        self.Wo = nn.Linear(emb_dim, output_dim)
        
        self.softmax = nn.Softmax(dim=-1)

    def expand(self, x):
        #y=x.repeat(x.size(0),x.size(1)*self.group_size,x.size(2),x.size(3))
        #y=x.repeat(1,self.group_size,1,1)
        x = x.unsqueeze(1)          # [B, 1, Hk, L, D]
        x = x.expand(-1, self.group_size , -1, -1, -1)  # [B, G, Hk, L, D]
        [B, G, Hk, L, D] = x.shape
        x=x.contiguous()
        x = x.view(B, G*Hk, L, D)  # [B, G*Hk, L, D]
        return x

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.Wq(x) # [B, L, D]
        K = self.Wk(x)
        V = self.Wv(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
        K_slim = K.view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1, 2)
        V_slim = V.view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1, 2)

        K=self.expand(K_slim)
        V=self.expand(V_slim)
        #print('Q:',Q.shape, 'K:',K.shape, 'V:',V.shape)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  
        attn_weights = self.softmax(attn_scores)
        
        z = torch.matmul(attn_weights, V)  
        
        z=z.transpose(1,2)
        z=z.contiguous()
        z=z.view(batch_size, seq_len, self.emb_dim) 
        z=z.contiguous()

        output = self.Wo(z)

        return output,K_slim,V_slim

net_MHA=MultiHeadAttention(emb_dim=256, num_heads=8,  output_dim=64)
net_MQA=MultiQueryAttention(emb_dim=256, num_heads=8, output_dim=64)
net_GQA=GroupedQueryAttention(emb_dim=256, num_heads=8,output_dim=64, group_size=4)

"""
比较3种方式输出的大小和KV cache的大小，符合预期
"""

a=torch.randn(5,6, 256)
output_MHA,K_MHA,Q_MHA = net_MHA(a)
output_MQA,K_MQA,Q_MQA = net_MQA(a)
output_GQA,K_GQA,Q_GQA = net_GQA(a)

print('MHA: output:',output_MHA.shape,' K:',K_MHA.shape,' V:',Q_MHA.shape)
print('MQA: output:',output_MQA.shape,' K:',K_MQA.shape,' V:',Q_MQA.shape)
print('GQA: output:',output_GQA.shape,' K:',K_GQA.shape,' V:',Q_GQA.shape)
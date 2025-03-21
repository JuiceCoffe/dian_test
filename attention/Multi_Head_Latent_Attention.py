import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[None, :, None, :]  
        cos = emb.cos()
        sin = emb.sin()
        return x * cos + self._rotate_half(x) * sin

class Multi_Head_Latent_Attention(nn.Module):
    def __init__(self,dim_emd, num_heads,dim_out,dim_c,dim_kqv,dim_r,dim_c_kv):
        super(Multi_Head_Latent_Attention, self).__init__()
        assert dim_kqv%num_heads==0, 'dim_kqv must be divided by num_heads'
        self.num_heads=num_heads

        self.W_c=nn.Linear(dim_emd,dim_c)#h->c
        self.W_cq=nn.Linear(dim_c,dim_kqv) #c->q
        self.W_qr=nn.Linear(dim_c,dim_r)#c->qr

        self.W_ckv=nn.Linear(dim_emd,dim_c_kv)#h->ckv
        self.W_ck=nn.Linear(dim_c_kv,dim_kqv)#ckv->k
        self.w_kr=nn.Linear(dim_emd,dim_r)#h->kr
        self.W_v=nn.Linear(dim_c_kv,dim_kqv)#ckv->v
        self.scaler=1/(dim_kqv//num_heads + dim_r//num_heads)**0.5
        self.w_o=nn.Linear(dim_kqv,dim_out)
        self.softmax=nn.Softmax(dim=-1)
        self.Rope = RotaryPositionalEmbedding(dim_r//num_heads)

        
    # def Rope(self,qr):
    #     return qr
        
    def forward(self, h_t):# x: [batch_size, seq_len, dim_emd]
        '''
        d_hr=dim_kqv//num_heads+dim_r//num_heads
        '''
        c=self.W_c(h_t)
        # print('c:',c.shape)
        q=self.W_cq(c)  #q: [batch_size, seq_len, dim_kqv]
        q=q.view(q.size(0),q.size(1),self.num_heads,-1)#q: [batch_size, seq_len, num_heads, dim_kqv//num_heads]
        qr=self.W_qr(c) #qr: [batch_size, seq_len, dim_r]
        qr=qr.view(qr.size(0),qr.size(1),self.num_heads,-1)#qr: [batch_size, seq_len, num_heads, dim_r//num_heads]
        qr=self.Rope(qr)
        q=torch.cat([q,qr],dim=-1)#q: [batch_size, seq_len, num_heads, dim_kqv//num_heads+dim_r//num_heads]

        c=self.W_ckv(h_t) #c: [batch_size, seq_len, dim_kqv]
        # print('c:',c.shape)
        k=self.W_ck(c) #k: [batch_size, seq_len, dim_kqv]
        k=k.view(k.size(0),k.size(1),self.num_heads,-1)#k: [batch_size, seq_len, num_heads, dim_kqv//num_heads]
        kr=self.w_kr(h_t) #kr: [batch_size, seq_len, dim_r]
        kr=kr.view(kr.size(0),kr.size(1),self.num_heads,-1)#kr: [batch_size, seq_len, num_heads, dim_r//num_heads]
        kr=self.Rope(kr)
        k=torch.cat([k,kr],dim=-1) #k: [batch_size, seq_len, num_heads, dim_kqv//num_heads+dim_r//num_heads]

        v=self.W_v(c) #v: [batch_size, seq_len, dim_kqv]
        output=torch.matmul(q,k.transpose(-2,-1))*self.scaler #output: [batch_size, seq_len, num_heads, num_heads]
        output=self.softmax(output)
        #print('v:',v.shape)
        v=v.view(v.size(0),v.size(1),self.num_heads,-1)#v: [batch_size, seq_len, num_heads, dim_kqv//num_heads]
        # print('output:',output.shape)
        # print('v:',v.shape)
        output=torch.matmul(output,v)#output: [batch_size, seq_len, num_heads, dim_kqv//num_heads]
        output=output.view(output.size(0),output.size(1),-1)#output: [batch_size, seq_len, dim_kqv]
        output=self.w_o(output)
        return output

net=Multi_Head_Latent_Attention(dim_emd=512, num_heads=8,dim_out=128,dim_c=256,dim_kqv=128,dim_c_kv=128,dim_r=64)
x=torch.randn(3,4,512)
y=net(x)
print(y.shape)

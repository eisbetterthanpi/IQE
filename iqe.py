# @title IQE me
# Improved Representation of Asymmetrical Distances with Interval Quasimetric Embeddings jan 2024 https://arxiv.org/pdf/2211.15120
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MaxMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(-torch.ones(1))

    def forward(self, d):
        a = torch.sigmoid(self.a)
        return a*d.max(dim=-1).values + (1-a)*d.mean(dim=-1)

class Reduction(nn.Module):
    def __init__(self):
        super().__init__()
        # self.red = lambda x: x.mean(-1)
        self.red = lambda x: x.sum(-1) # IQE-sum
        # self.red = lambda x: x.max(-1).values
        # self.red = MaxMean() # IQE-maxmean

    def forward(self, d):
        # return self.red(d)
        return .9**self.red(d) # γ-discount

def U_interval(u, v): # d(u,v) = |Union_i [u_i, max(u_i, v_i)]|
    end = torch.max(u, v)
    order = torch.argsort(u, dim=-1)
    u, end = torch.gather(u, -1, order), torch.gather(end, -1, order)
    running_end = torch.cat([u[...,:1], torch.cummax(end[...,:-1], dim=-1).values], dim=-1)
    return torch.clamp(end - torch.max(u, running_end), min=0).sum(-1)

class IQE(nn.Module):
    def __init__(self, d_head=16):
        super().__init__()
        self.d_head = d_head
        self.reduction = Reduction()

    def forward(self, x, y): # [...,d]
        x, y = x.unflatten(-1, (-1,self.d_head)), y.unflatten(-1, (-1,self.d_head))
        d = U_interval(x,y)
        return self.reduction(d) # [...]


b,d = 2,128
d_head=16
d_iqe = IQE(d_head).to(device) # split 128 dimensions into 16-dimenional chunks, where each chunk gives an IQE component (IQE paper recommends `dim_per_component >= 8`)
# latents, usually from an encoder. use random vectors as example
x = torch.randn(b,d, device=device)
y = torch.randn(b,d, device=device)

print(d_iqe(x, y)) # distance # [...,d]->[...]
# print(d_iqe(x[:, None], y)) # cdist # bbd
# print(d_iqe(x[:, None], x)) # pdist # bbd


# optim = torch.optim.AdamW(d_iqe.parameters(), lr=1e-1)
# for _ in range(10):
#     loss = ((1-d_iqe(x, y))**2).mean()
#     print(loss)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()


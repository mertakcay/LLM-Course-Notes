import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import CausalSelfAttention

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_pdrop)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Config:
    def __init__(self, vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
                 block_size=1024, bias=True, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.bias = bias
        self.embd_pdrop = dropout
        self.resid_pdrop = dropout
        self.attn_pdrop = dropout

class GPT2SmallConfig(GPT2Config):
    def __init__(self):
        super().__init__(n_layer=12, n_head=12, n_embd=768)

class GPT2MediumConfig(GPT2Config):
    def __init__(self):
        super().__init__(n_layer=24, n_head=16, n_embd=1024)

class GPT2LargeConfig(GPT2Config):
    def __init__(self):
        super().__init__(n_layer=36, n_head=20, n_embd=1280)

class GPT2XLConfig(GPT2Config):
    def __init__(self):
        super().__init__(n_layer=48, n_head=25, n_embd=1600)

class GPT2TinyConfig(GPT2Config):
    def __init__(self):
        super().__init__(n_layer=6, n_head=6, n_embd=384)

class GPT2NanoConfig(GPT2Config):
    def __init__(self):
        super().__init__(n_layer=3, n_head=3, n_embd=192)
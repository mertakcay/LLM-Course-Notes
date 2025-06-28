import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    def forward(self, x):
        batch_size, seq_len, n_embd = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.query_layers = nn.ModuleList([nn.Linear(config.n_embd, self.head_size, bias=config.bias) for _ in range(config.n_head)])
        self.key_layers = nn.ModuleList([nn.Linear(config.n_embd, self.head_size, bias=config.bias) for _ in range(config.n_head)])
        self.value_layers = nn.ModuleList([nn.Linear(config.n_embd, self.head_size, bias=config.bias) for _ in range(config.n_head)])
        self.output_projection = nn.Linear(config.n_head * self.head_size, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, config.block_size, config.block_size)
        )
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        head_outputs = []
        for head_idx in range(self.n_head):
            queries = self.query_layers[head_idx](x)
            keys = self.key_layers[head_idx](x)
            values = self.value_layers[head_idx](x)
            scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.head_size)
            scores = scores.masked_fill(self.mask[:, :seq_len, :seq_len] == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            head_output = torch.bmm(attn_weights, values)
            head_outputs.append(head_output)
        multi_head_output = torch.cat(head_outputs, dim=-1)
        output = self.resid_dropout(self.output_projection(multi_head_output))
        return output
import torch
import torch.nn as nn
from .attention import MosaicBertAttention

class MosaicBertFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MosaicBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MosaicBertAttention(config)
        self.feed_forward = MosaicBertFeedForward(config)
        
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.LayerNorm1(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + self.dropout1(attention_output)
        
        residual = hidden_states
        hidden_states = self.LayerNorm2(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout2(feed_forward_output)
        
        return hidden_states

class MosaicBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([MosaicBertLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
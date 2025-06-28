import torch
import torch.nn as nn
from .attention import ModernBertAttention

class ModernBertFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_fn = nn.GELU()
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states):
        hidden_states_normalized = self.pre_layer_norm(hidden_states)
        
        hidden_states_1 = self.dense_1(hidden_states_normalized)
        hidden_states_gate = self.act_fn(self.gate(hidden_states_normalized))
        hidden_states = hidden_states_1 * hidden_states_gate
        
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states

class ModernBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ModernBertAttention(config)
        self.feed_forward = ModernBertFeedForward(config)
        
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout1(attention_output)
        
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = hidden_states + self.dropout2(feed_forward_output)
        
        return hidden_states

class ModernBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([ModernBertLayer(config) for _ in range(config.num_hidden_layers)])
        
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states
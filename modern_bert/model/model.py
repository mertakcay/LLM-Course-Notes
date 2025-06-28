import torch
import torch.nn as nn
from .embeddings import ModernBertEmbeddings
from .transformer import ModernBertEncoder

class ModernBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = ModernBertEmbeddings(config)
        self.encoder = ModernBertEncoder(config)
        
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None):
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
            
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        sequence_output = self.encoder(embedding_output, extended_attention_mask)
        
        pooled_output = self.pooler(sequence_output[:, 0])
        
        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output
        }

class ModernBertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = ModernBertModel(config)
        
        self.cls = ModernBertPreTrainingHeads(config)
        
        self.apply(self._init_weights)
        
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']
        
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        result = {
            'prediction_logits': prediction_scores,
            'seq_relationship_logits': seq_relationship_score
        }
        
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            result['loss'] = total_loss
            
        return result

class ModernBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = ModernBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class ModernBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
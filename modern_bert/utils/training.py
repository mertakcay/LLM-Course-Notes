import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random
import tqdm

class MaskedLanguageModelingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, mlm_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        input_ids = encoding['input_ids']
        token_type_ids = encoding['token_type_ids']
        attention_mask = encoding['attention_mask']
        
        masked_input_ids, mlm_labels = self.mask_tokens(input_ids, attention_mask)
        
        return {
            'input_ids': torch.tensor(masked_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long)
        }
    
    def mask_tokens(self, input_ids, attention_mask):
        labels = input_ids.copy()
        
        probability_matrix = np.full(len(input_ids), self.mlm_probability)
        
        special_tokens_mask = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        ]
        for i, token_id in enumerate(input_ids):
            if token_id in special_tokens_mask or attention_mask[i] == 0:
                probability_matrix[i] = 0.0
        
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels = [-100 if not masked else token for masked, token in zip(masked_indices, labels)]
        
        indices_replaced = np.random.binomial(1, 0.8, size=len(input_ids)).astype(bool) & masked_indices
        for i in np.where(indices_replaced)[0]:
            input_ids[i] = self.tokenizer.mask_token_id
        
        indices_random = np.random.binomial(1, 0.5, size=len(input_ids)).astype(bool) & masked_indices & ~indices_replaced
        for i in np.where(indices_random)[0]:
            input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
        
        return input_ids, labels

class NextSentencePredictionDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, max_length=512):
        self.sentence_pairs = sentence_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentence_pairs)
    
    def __getitem__(self, idx):
        sentence_a, sentence_b, is_next = self.sentence_pairs[idx]
        
        encoding = self.tokenizer.encode(
            sentence_a,
            sentence_b,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        input_ids = encoding['input_ids']
        token_type_ids = encoding['token_type_ids']
        attention_mask = encoding['attention_mask']
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'next_sentence_label': torch.tensor(is_next, dtype=torch.long)
        }

class ModernBertTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        optimizer=None,
        lr_scheduler=None,
        batch_size=16,
        num_epochs=3,
        device=None,
        checkpoint_path=None,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        use_amp=False,
        warmup_steps=0,
        logging_steps=100,
        eval_steps=None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if optimizer is None:
            self.optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        else:
            self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        
        self.checkpoint_path = checkpoint_path
        
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        
        if self.lr_scheduler is None and self.warmup_steps > 0:
            from transformers import get_linear_schedule_with_warmup
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
        
        self.model.train()
        global_step = 0
        epoch_loss = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            epoch_loss = 0.0
            
            progress_bar = tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch+1}")
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with automatic mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'],
                            masked_lm_labels=batch.get('mlm_labels'),
                            next_sentence_label=batch.get('next_sentence_label')
                        )
                        loss = outputs['loss'] if 'loss' in outputs else None
                        loss = loss / self.gradient_accumulation_steps  # Scale loss for gradient accumulation
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        masked_lm_labels=batch.get('mlm_labels'),
                        next_sentence_label=batch.get('next_sentence_label')
                    )
                    loss = outputs['loss'] if 'loss' in outputs else None
                    loss = loss / self.gradient_accumulation_steps if loss is not None else None  # Scale loss for gradient accumulation
                
                # Backward pass with automatic mixed precision
                if loss is not None:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Update weights
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Update learning rate
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        print(f"Step {global_step}/{total_steps}, Loss: {epoch_loss / (step + 1):.4f}")
                    
                    # Evaluation
                    if self.eval_dataset is not None and self.eval_steps is not None and global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate()
                        print(f"Evaluation at step {global_step}: Loss: {eval_loss:.4f}")
                        self.model.train()  # Set model back to training mode
                
                # Update progress bar
                progress_bar.set_postfix({'loss': epoch_loss / (step + 1)})
            
            # Evaluate at the end of each epoch
            if self.eval_dataset is not None:
                eval_loss = self.evaluate()
                print(f"Evaluation at epoch {epoch+1}: Loss: {eval_loss:.4f}")
            
            # Save checkpoint
            if self.checkpoint_path is not None:
                self.save_checkpoint(epoch, global_step)
    
    def evaluate(self):
        """Evaluate the model with modern evaluation techniques."""
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size
        )
        
        # Evaluation loop
        self.model.eval()
        eval_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with automatic mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'],
                            masked_lm_labels=batch.get('mlm_labels'),
                            next_sentence_label=batch.get('next_sentence_label')
                        )
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        masked_lm_labels=batch.get('mlm_labels'),
                        next_sentence_label=batch.get('next_sentence_label')
                    )
                
                loss = outputs['loss'] if 'loss' in outputs else None
                
                if loss is not None:
                    eval_loss += loss.item()
        
        return eval_loss / len(eval_dataloader)
    
    def save_checkpoint(self, epoch, global_step):
        """Save model checkpoint with modern metadata."""
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, f"{self.checkpoint_path}/checkpoint_epoch_{epoch}_step_{global_step}.pt")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint with modern metadata."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch'], checkpoint.get('global_step', 0)
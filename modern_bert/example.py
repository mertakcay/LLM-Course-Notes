import torch
import torch.nn as nn
import torch.optim as optim
from model.model import ModernBertForPreTraining
from config import ModernBertConfig
from utils.tokenization import ModernBertTokenizer
from utils.training import MaskedLanguageModelingDataset, NextSentencePredictionDataset, ModernBertTrainer
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Modern BERT")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--vocab_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(file_path):
    texts = ["Example text for masked language modeling."]
    sentence_pairs = [("First sentence.", "Second sentence.", 1)]
    return texts, sentence_pairs

def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model_size == "base":
        config = ModernBertConfig.get_base_config()
    else:
        config = ModernBertConfig.get_large_config()
    
    tokenizer = ModernBertTokenizer(vocab_file=args.vocab_file)
    
    model = ModernBertForPreTraining(config)
    
    train_texts, train_sentence_pairs = load_data(args.train_file)
    eval_texts, eval_sentence_pairs = load_data(args.eval_file) if args.eval_file else (None, None)
    
    train_mlm_dataset = MaskedLanguageModelingDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        mlm_probability=args.mlm_probability
    )
    
    train_nsp_dataset = NextSentencePredictionDataset(
        sentence_pairs=train_sentence_pairs,
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    
    eval_dataset = None
    if args.eval_file:
        eval_mlm_dataset = MaskedLanguageModelingDataset(
            texts=eval_texts,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            mlm_probability=args.mlm_probability
        )
        eval_dataset = eval_mlm_dataset
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Initialize trainer with modern features
    trainer = ModernBertTrainer(
        model=model,
        train_dataset=train_mlm_dataset,  # You can switch between MLM and NSP datasets
        eval_dataset=eval_dataset,
        optimizer=optimizer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        checkpoint_path=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        warmup_steps=args.warmup_steps,
        logging_steps=100,
        eval_steps=1000
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "modern_bert_final.pt"))
    
    print("Training complete!")

def fine_tune_example():
    """Example of how to fine-tune Modern BERT for a specific task."""
    # Load pre-trained model
    config = ModernBertConfig.get_base_config()
    model = ModernBertForPreTraining(config)
    model.load_state_dict(torch.load("path/to/pretrained/model.pt"))
    
    # Create task-specific head (e.g., for classification)
    class ModernBertForSequenceClassification(nn.Module):
        def __init__(self, bert_model, num_labels, dropout_prob=0.1):
            super().__init__()
            self.bert = bert_model.bert  # Use the base BERT model without pre-training heads
            self.dropout = nn.Dropout(dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, num_labels)
            
            # Modern approach: initialize classifier weights
            with torch.no_grad():
                self.classifier.weight.normal_(mean=0.0, std=0.02)
                self.classifier.bias.zero_()
        
        def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            pooled_output = outputs['pooler_output']
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            
            return {
                'loss': loss,
                'logits': logits
            }
    
    # Create classification model
    num_labels = 2  # Binary classification
    classification_model = ModernBertForSequenceClassification(model, num_labels)
    
    # Modern fine-tuning approach with layer-wise learning rate decay
    # Higher learning rates for task-specific layers, lower for base layers
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in classification_model.classifier.named_parameters()],
            "lr": 5e-5,
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in classification_model.bert.named_parameters()],
            "lr": 1e-5,  # Lower learning rate for pre-trained parameters
            "weight_decay": 0.01,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    # Use learning rate scheduler
    num_training_steps = 1000  # Example value
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=5e-5, 
        total_steps=num_training_steps
    )
    
    # Example training loop with modern techniques
    classification_model.train()
    for epoch in range(3):
        # Training code would go here
        # Example of using mixed precision
        scaler = torch.cuda.amp.GradScaler()
        for batch in range(10):  # Placeholder for actual batches
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                # Assume batch_data contains input_ids, attention_mask, etc.
                outputs = classification_model(
                    input_ids=None,  # Replace with actual batch data
                    attention_mask=None,  # Replace with actual batch data
                    labels=None  # Replace with actual batch data
                )
                loss = outputs['loss']
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update learning rate
            lr_scheduler.step()
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
    # Uncomment to run fine-tuning example
    # fine_tune_example()
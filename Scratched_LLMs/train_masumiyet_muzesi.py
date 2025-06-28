import os
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.transformer import GPT2Config, GPT2TinyConfig, GPT2NanoConfig
from model.gpt2 import GPT2
from utils.tokenizer import BPETokenizer
from utils.data_utils import load_dataset, create_dataloaders


def load_or_create_tokenizer(tokenizer_path=None, train_dir=None, vocab_size=30000):
    if tokenizer_path and os.path.exists(tokenizer_path):
        return BPETokenizer.load(tokenizer_path)
    elif train_dir:
        return train_tokenizer(train_dir, vocab_size)
    else:
        raise ValueError("Either tokenizer_path or train_dir must be provided")


def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 on Masumiyet Müzesi text')
    parser.add_argument('--data_dir', type=str, default='/Users/mertakcay/Documents/LLM/data/text', 
                        help='Directory containing the text data')
    parser.add_argument('--output_dir', type=str, default='/Users/mertakcay/Documents/LLM/output', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--model_size', type=str, default='nano', choices=['nano', 'tiny', 'small'], 
                        help='Model size to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--block_size', type=int, default=128, help='Context size for training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=50, help='Number of warmup steps for LR scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--eval_interval', type=int, default=200, help='Interval to evaluate model')
    parser.add_argument('--save_interval', type=int, default=500, help='Interval to save model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to train on')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size for tokenizer')
    return parser.parse_args()


def get_model_config(model_size, vocab_size, block_size):
    if model_size == 'nano':
        config = GPT2NanoConfig()
    elif model_size == 'tiny':
        config = GPT2TinyConfig()
    elif model_size == 'small':
        config = GPT2Config(n_layer=12, n_head=12, n_embd=768)
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    config.vocab_size = vocab_size
    config.block_size = block_size
    return config


def train_tokenizer(data_dir, vocab_size):
    print("Training tokenizer...")
    all_texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            all_texts.append(text)
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(all_texts)
    os.makedirs('output', exist_ok=True)
    tokenizer.save('output/tokenizer_masumiyet.json')
    return tokenizer


def train_model(args):
    device = args.device
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer_masumiyet.json")
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = load_or_create_tokenizer(tokenizer_path=tokenizer_path)
    else:
        print("Training new tokenizer on Masumiyet Müzesi text")
        tokenizer = train_tokenizer(args.data_dir, args.vocab_size)
    print("Loading and tokenizing dataset...")
    train_dataset, val_dataset = load_dataset(args.data_dir, tokenizer, block_size=args.block_size)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=args.batch_size)
    print(f"Initializing {args.model_size} GPT-2 model...")
    config = get_model_config(args.model_size, tokenizer.vocab_size, args.block_size)
    model = GPT2(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    print(f"Starting training for {args.epochs} epochs...")
    global_step = 0
    total_steps = args.epochs * len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_item = loss.item()
            epoch_losses.append(loss_item)
            progress_bar.set_postfix({"loss": f"{loss_item:.4f}"})
            global_step += 1
            if global_step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                val_losses.append(val_loss)
                train_losses.append(sum(epoch_losses) / len(epoch_losses))
                print(f"Step {global_step}: Train loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_masumiyet.pt"))
                    print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                model.train()
            if global_step % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt"))
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{args.epochs} completed, Average Loss: {avg_epoch_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model_masumiyet.pt"))
    print(f"Final model saved to {os.path.join(args.output_dir, 'final_model_masumiyet.pt')}")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'training_stats_masumiyet.png'))
    return model, tokenizer


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    
    return total_loss / total_samples


def generate_sample(model, tokenizer, prompt="Hayatımın", max_tokens=100, temperature=0.8, device="cpu"):
    """Generate a sample text from the model"""
    model.eval()
    model = model.to(device)
    
    # Encode the prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
    
    # Decode the output
    generated_text = tokenizer.decode(output[0].tolist())
    
    return generated_text


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = train_model(args)
    
    # Generate a sample
    sample = generate_sample(model, tokenizer, prompt="Hayatımın en mutlu anı", device=args.device)
    print("\nGenerated sample:")
    print(sample)
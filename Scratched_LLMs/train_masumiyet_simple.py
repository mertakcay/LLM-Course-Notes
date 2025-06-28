import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, max_seq_length=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=dim_feedforward, 
                                                 dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
    def forward(self, x, targets=None):
        x = x[:, :self.max_seq_length]
        seq_length = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :seq_length, :]
        x = self.transformer(x)
        logits = self.output(x)
        loss = None
        if targets is not None:
            targets = targets[:, :seq_length]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_length else idx[:, -self.max_seq_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    def fit(self, text):
        unique_chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        return self
    def encode(self, text):
        return [self.char_to_idx.get(char, 0) for char in text]
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = tokenizer.encode(text)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y

def train_on_masumiyet_muzesi():
    seq_length = 64
    batch_size = 32
    epochs = 3
    learning_rate = 3e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    text_path = "/Users/mertakcay/Documents/LLM/data/text/masumiyet_muzesi.txt"
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer().fit(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    dataset = TextDataset(text, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} sequences")
    model = TinyGPT(vocab_size=tokenizer.vocab_size, max_seq_length=seq_length)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            if batch_idx % 100 == 0 and batch_idx > 0:
                model.eval()
                with torch.no_grad():
                    prompt = "Hayatımın"
                    prompt_encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
                    generated = model.generate(prompt_encoded, max_new_tokens=50, temperature=0.8)
                    generated_text = tokenizer.decode(generated[0].tolist())
                    print(f"\nSample: {generated_text}\n")
                model.train()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}")
    output_dir = "/Users/mertakcay/Documents/LLM/output"
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': {
            'char_to_idx': tokenizer.char_to_idx,
            'idx_to_char': tokenizer.idx_to_char,
            'vocab_size': tokenizer.vocab_size
        }
    }, os.path.join(output_dir, "masumiyet_model.pt"))
    print(f"Model saved to {os.path.join(output_dir, 'masumiyet_model.pt')}")
    model.eval()
    with torch.no_grad():
        prompt = "Hayatımın en mutlu anı"
        prompt_encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        generated = model.generate(prompt_encoded, max_new_tokens=100, temperature=0.8)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"\nFinal sample:\n{generated_text}\n")
    return model, tokenizer

if __name__ == "__main__":
    train_on_masumiyet_muzesi()
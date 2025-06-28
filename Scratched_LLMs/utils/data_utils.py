import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_dataset(data_dir, tokenizer, block_size=1024, split_ratio=0.9):
    all_texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            all_texts.append(text)
    combined_text = tokenizer.encode('\n'.join(all_texts))
    total_tokens = len(combined_text)
    if block_size >= total_tokens - 1:
        block_size = max(1, total_tokens // 2 - 1)
        print(f"Adjusting block_size to {block_size} due to small dataset size")
    min_required = (block_size + 1) * 2
    if total_tokens < min_required:
        print(f"Warning: Dataset too small ({total_tokens} tokens) for splitting. Using all data for training.")
        train_data = combined_text
        val_data = combined_text[:block_size+1]
    else:
        train_size = int(len(combined_text) * split_ratio)
        train_size = max(train_size, block_size + 1)
        train_size = min(train_size, len(combined_text) - (block_size + 1))
        train_data = combined_text[:train_size]
        val_data = combined_text[train_size:]
    print(f"Train data size: {len(train_data)} tokens")
    print(f"Validation data size: {len(val_data)} tokens")
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size=4, num_workers=0):
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            f"Dataset too small for the block size. Train size: {len(train_dataset)}, "
            f"Val size: {len(val_dataset)}. Try reducing block_size or using more data."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def prepare_orhan_pamuk_dataset(data_dir, tokenizer, block_size=1024, batch_size=4):
    print("Loading and tokenizing Orhan Pamuk dataset...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    all_texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            all_texts.append(text)
    combined_text = tokenizer.encode('\n'.join(all_texts))
    total_tokens = len(combined_text)
    adjusted_block_size = block_size
    if total_tokens < block_size * 3:
        adjusted_block_size = max(1, total_tokens // 3 - 1)
        print(f"Warning: Dataset is small ({total_tokens} tokens). Reducing block_size from {block_size} to {adjusted_block_size}")
    train_dataset, val_dataset = load_dataset(data_dir, tokenizer, adjusted_block_size)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size)
    return train_loader, val_loader


def get_sample_batch(dataloader):
    for batch in dataloader:
        return batch


def save_dataset_stats(data_dir, output_file):
    stats = {
        "num_files": 0,
        "total_size": 0,
        "total_lines": 0,
        "files": []
    }
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            file_size = os.path.getsize(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                num_lines = len(lines)
            stats["num_files"] += 1
            stats["total_size"] += file_size
            stats["total_lines"] += num_lines
            stats["files"].append({
                "name": filename,
                "size": file_size,
                "lines": num_lines
            })
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset Statistics\n")
        f.write(f"=================\n\n")
        f.write(f"Number of files: {stats['num_files']}\n")
        f.write(f"Total size: {stats['total_size'] / 1024 / 1024:.2f} MB\n")
        f.write(f"Total lines: {stats['total_lines']}\n\n")
        f.write(f"Files:\n")
        for file_info in stats["files"]:
            f.write(f"- {file_info['name']}: {file_info['size'] / 1024:.2f} KB, {file_info['lines']} lines\n")
    return stats
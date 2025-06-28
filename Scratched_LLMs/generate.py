import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from model.transformer import GPT2Config, GPT2TinyConfig, GPT2NanoConfig
from model.gpt2 import GPT2
from utils.tokenizer import BPETokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Generate text using trained GPT-2 model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='output/tokenizer.json', help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='', help='Prompt to start generation with')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to generate on')
    parser.add_argument('--model_size', type=str, default='nano', choices=['nano', 'tiny', 'small'], 
                        help='Model size to load')
    parser.add_argument('--output_file', type=str, default=None, help='File to save generated text to')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    return parser.parse_args()


def get_model_config(model_size, vocab_size, block_size=1024):
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


def load_model(model_path, model_size, vocab_size, device):
    config = get_model_config(model_size, vocab_size)
    model = GPT2(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, max_tokens, temperature, top_k, device):
    if prompt:
        prompt_tokens = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
    else:
        prompt_tensor = torch.zeros((1, 1), dtype=torch.long).to(device)
    with torch.no_grad():
        generated = model.generate(
            prompt_tensor, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = BPETokenizer.load(args.tokenizer_path)
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.model_size, tokenizer.vocab_size, device)
    print("\nGenerating text...")
    all_samples = []
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {i+1}/{args.num_samples}:")
        generated_text = generate_text(
            model, tokenizer, args.prompt, args.max_tokens, 
            args.temperature, args.top_k, device
        )
        if args.prompt:
            print(f"Prompt: {args.prompt}")
        print(f"Generated: {generated_text}")
        all_samples.append(generated_text)
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(all_samples):
                f.write(f"Sample {i+1}:\n")
                f.write(f"{sample}\n\n")
        print(f"\nGenerated text saved to {args.output_file}")


if __name__ == '__main__':
    main()
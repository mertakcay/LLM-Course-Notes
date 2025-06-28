import torch
from torch.utils.data import Dataset, DataLoader
import random
import config
from tokenizer import BPETokenizer
import os

class MLMDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vocab_size = len(tokenizer.token_to_id)
        self.pad_token_id = tokenizer.token_to_id.get(config.SPECIAL_TOKENS[0], 0)
        self.mask_token_id = tokenizer.token_to_id.get(config.SPECIAL_TOKENS[4], -1)
        self.cls_token_id = tokenizer.token_to_id.get(config.SPECIAL_TOKENS[2], -1)
        self.sep_token_id = tokenizer.token_to_id.get(config.SPECIAL_TOKENS[3], -1)

        if self.mask_token_id == -1 or self.cls_token_id == -1 or self.sep_token_id == -1:
            self.pad_token_id = config.PAD_TOKEN_ID
            self.mask_token_id = config.MASK_TOKEN_ID
            self.cls_token_id = config.CLS_TOKEN_ID
            self.sep_token_id = config.SEP_TOKEN_ID
            if self.mask_token_id == -1 or self.cls_token_id == -1 or self.sep_token_id == -1:
                 raise ValueError(
                    f"[MASK], [CLS], or [SEP] token not found in tokenizer vocabulary or config. "
                    f"Tokenizer vocab: {tokenizer.token_to_id}. "
                    f"Expected special tokens: {config.SPECIAL_TOKENS}"
                )
        self.lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip())
        
        self.vocab_items = list(self.tokenizer.token_to_id.keys())


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        
        raw_tokens = self.tokenizer.tokenize(line)

        input_ids, labels = self._mask_tokens(raw_tokens)

        input_ids = [config.CLS_TOKEN_ID] + input_ids + [config.SEP_TOKEN_ID]
        labels = [-100] + labels + [-100]

        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
        
        attention_mask = [1] * len(input_ids)
        
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [config.PAD_TOKEN_ID] * padding_length
        labels += [-100] * padding_length
        attention_mask += [0] * padding_length
        
        token_type_ids = [0] * self.max_seq_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def _mask_tokens(self, token_ids):
        labels = [-100] * len(token_ids)
        output_token_ids = list(token_ids)

        mask_probability = getattr(config, 'MASK_PROB', 0.15)

        for i, token_id in enumerate(token_ids):
            prob = random.random()
            if prob < mask_probability:
                labels[i] = token_id

                mask_decision_prob = random.random()
                if mask_decision_prob < 0.8:
                    output_token_ids[i] = config.MASK_TOKEN_ID
                elif mask_decision_prob < 0.9:
                    random_token_id = random.choice(list(self.tokenizer.id_to_token.keys()))
                    output_token_ids[i] = random_token_id
        
        return output_token_ids, labels


if __name__ == '__main__':
    dummy_corpus_path = config.CORPUS_FILE_PATH
    try:
        tokenizer = BPETokenizer.load_tokenizer(config.TOKENIZER_PATH)
        print("Loaded existing tokenizer.")
    except FileNotFoundError:
        print("Tokenizer not found. Training a dummy one.")
        if not os.path.exists(dummy_corpus_path):
            print(f"Creating dummy corpus at {dummy_corpus_path}")
            with open(dummy_corpus_path, 'w', encoding='utf-8') as f:
                f.write("This is a test sentence for the dataset.\n")
                f.write("Another example sentence for masking.\n")
        else:
            print(f"Using existing dummy corpus at {dummy_corpus_path}")

        tokenizer = BPETokenizer(vocab_size=config.VOCAB_SIZE, special_tokens=config.SPECIAL_TOKENS)
        tokenizer.train(dummy_corpus_path)
        tokenizer.save_tokenizer(config.TOKENIZER_PATH)
        print(f"Dummy tokenizer trained and saved to {config.TOKENIZER_PATH}")

    print(f"Using vocab size: {len(tokenizer.token_to_id)}")
    mask_token_id_to_print = tokenizer.token_to_id.get("[MASK]", config.MASK_TOKEN_ID)
    pad_token_id_to_print = tokenizer.token_to_id.get("[PAD]", config.PAD_TOKEN_ID)
    print(f"MASK_TOKEN_ID: {mask_token_id_to_print}, PAD_TOKEN_ID: {pad_token_id_to_print}")


    dataset = MLMDataset(file_path=config.CORPUS_FILE_PATH, tokenizer=tokenizer, max_seq_length=config.MAX_POSITION_EMBEDDINGS)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample from MLMDataset:")
        for key, value in sample.items():
            print(f"{key}: {value.shape} - {value}")

        pad_id_for_check = tokenizer.token_to_id.get("[PAD]", config.PAD_TOKEN_ID)
        original_tokens = [tokenizer.id_to_token.get(tid.item(), "[UNK]") for tid in sample['input_ids'] if tid.item() != pad_id_for_check]
        label_tokens = []
        for i, lab_id in enumerate(sample['labels']):
            if lab_id.item() != -100:
                original_for_masked = tokenizer.id_to_token.get(lab_id.item(), "[ERR]")
                current_token_in_input = tokenizer.id_to_token.get(sample['input_ids'][i].item(), "[ERR]")
                label_tokens.append(f"({current_token_in_input} -> {original_for_masked})")
            else:
                label_tokens.append(tokenizer.id_to_token.get(sample['input_ids'][i].item(), "[UNK]"))
        
        print("\nVisualizing masking (approximate):")
        pad_token_str = tokenizer.id_to_token.get(pad_id_for_check, "[PAD]")
        print("Input: ", " ".join(tk for tk in label_tokens if tk != pad_token_str))

        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE)
        batch = next(iter(dataloader))
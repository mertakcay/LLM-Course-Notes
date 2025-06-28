import os
import json
import regex as re
import torch
import numpy as np
from collections import defaultdict

class Tokenizer:
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<|endoftext|>': 50256,
        }
        self.initialized = False
    def train(self, texts, vocab_size=None):
        if vocab_size is not None:
            self.vocab_size = vocab_size
        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        counts = defaultdict(int)
        for text in texts:
            for char in text:
                if ord(char) < 256:
                    char = byte_encoder[ord(char)]
                counts[char] += 1
        vocab = sorted(counts.items(), key=lambda x: -x[1])
        vocab = [x[0] for x in vocab]
        merges = []
        for i in range(self.vocab_size - len(vocab) - len(self.special_tokens)):
            if not vocab:
                break
            pairs = self._get_pairs(vocab)
            if not pairs:
                break
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            merges.append(best_pair)
            vocab = self._merge_pair(vocab, best_pair)
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.merges = merges
        self.initialized = True
        return self
    def _get_pairs(self, vocab):
        pairs = defaultdict(int)
        for word in vocab:
            chars = list(word)
            for i in range(len(chars) - 1):
                pairs[(chars[i], chars[i+1])] += 1
        return pairs
    def _merge_pair(self, vocab, pair):
        first, second = pair
        merged = first + second
        result = []
        for word in vocab:
            i = 0
            new_word = []
            chars = list(word)
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == first and chars[i+1] == second:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(chars[i])
                    i += 1
            result.append(''.join(new_word))
        return result
    def encode(self, text):
        if not self.initialized:
            raise ValueError("Tokenizer must be trained before encoding")
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.token_to_id.get('<|unk|>', 0))
        return tokens
    def decode(self, ids):
        if not self.initialized:
            raise ValueError("Tokenizer must be trained before decoding")
        text = ''
        for id in ids:
            if id in self.id_to_token:
                text += self.id_to_token[id]
        return text
    def save(self, path):
        if not self.initialized:
            raise ValueError("Tokenizer must be trained before saving")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'merges': self.merges,
                'special_tokens': self.special_tokens,
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False)
    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(i): token for i, token in tokenizer.token_to_id.items()}
        tokenizer.merges = data['merges']
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.initialized = True
        return tokenizer


class BPETokenizer:
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r'\s+|\s*([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])\s*|\S+|\s+', re.UNICODE)
        self.cache = {}
        self.initialized = False
    def train(self, texts, vocab_size=None):
        if vocab_size is not None:
            self.vocab_size = vocab_size
        corpus = ' '.join(texts)
        corpus_bytes = corpus.encode('utf-8', errors='replace')
        corpus_unicode = ''.join([self.byte_encoder[b] for b in corpus_bytes])
        words = re.findall(r'\S+|[^\S\r\n]+|[\r\n]', corpus_unicode)
        word_counts = defaultdict(int)
        for word in words:
            word_with_spaces = ' '.join(word)
            word_counts[word_with_spaces] += 1
        chars = set()
        for word in word_counts:
            for char in word.split():
                chars.add(char)
        vocab = sorted(list(chars))
        num_merges = min(self.vocab_size - len(vocab) - 2, 20000)
        merges = []
        print(f"Starting BPE training with {num_merges} merges")
        
        # Iteratively perform merges
        for i in range(num_merges):
            # Get statistics on pairs
            pairs = get_stats(word_counts)
            if not pairs or len(pairs) == 0:
                print(f"No more pairs to merge after {i} merges")
                break
                
            # Find the most frequent pair
            best = max(pairs, key=pairs.get)
            best_freq = pairs[best]
            
            # Print progress occasionally
            if i % 100 == 0:
                print(f"Merge #{i}: {best} (frequency: {best_freq})")
                
            # Add the merge to our list
            merges.append(best)
            
            # Update the vocabulary with the merged pair
            first, second = best
            new_token = first + second
            
            # Merge the pair in all words
            new_word_counts = {}
            for word, count in word_counts.items():
                # Use the global merge function
                new_word = globals()['merge'](word, first, second)
                new_word_counts[new_word] = count
            word_counts = new_word_counts
        
        # Create vocabulary with BPE merges
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        # Build encoder and decoder
        self.encoder = {}
        
        # Add all individual characters to the vocabulary
        for token in vocab:
            self.encoder[token] = len(self.encoder)
        
        # Add all merged tokens to the vocabulary
        for merge in merges:
            token = ''.join(merge)
            self.encoder[token] = len(self.encoder)
        
        # Add special tokens
        self.encoder['<|endoftext|>'] = len(self.encoder)
        self.encoder['<|unk|>'] = len(self.encoder)  # Add unknown token
        
        # Create decoder (id -> token mapping)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        print(f"Tokenizer trained with {len(self.encoder)} tokens and {len(self.bpe_ranks)} BPE merges")
        self.initialized = True
        
        return self
    
    def bpe(self, token):
        """Apply BPE encoding to a token"""
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
                
            pairs = get_pairs(word)
        
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    def _merge_pair(self, word_counts, pair):
        first, second = pair
        merged = first + second
        new_word_counts = defaultdict(int)
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word, count in word_counts.items():
            new_word = p.sub(merged, word)
            new_word_counts[new_word] += count
        return new_word_counts
    
    def encode(self, text):
        text = self._normalize_turkish_chars(text)
        words = re.findall(self.pat, text)
        tokens = []
        for word in words:
            if word in self.cache:
                tokens.extend(self.cache[word])
                continue
            word_tokens = self._bpe(word)
            ids = [self.encoder[token] for token in word_tokens]
            self.cache[word] = ids
            tokens.extend(ids)
        return tokens
    
    def decode(self, ids):
        if not self.initialized:
            raise ValueError("Tokenizer must be trained before decoding")
        
        # Handle empty list case
        if not ids:
            return ""
            
        # Convert token IDs to tokens
        tokens = []
        for id in ids:
            if id in self.decoder:
                tokens.append(self.decoder[id])
            else:
                print(f"Warning: Unknown token ID: {id}")
                # Skip unknown token IDs
                continue
        
        # Join tokens into a single string
        text = ''.join(tokens)
        
        # Convert from byte-level encoding back to UTF-8
        try:
            # Convert each character to its byte representation
            bytes_data = bytearray()
            i = 0
            while i < len(text):
                char = text[i]
                if char in self.byte_decoder:
                    bytes_data.append(self.byte_decoder[char])
                i += 1
            
            # Decode bytes to UTF-8 string
            return bytes_data.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Error decoding text: {e}")
            return text  # Return the raw text as fallback
    
    def save(self, path):
        """Save the tokenizer to a file"""
        if not self.initialized:
            raise ValueError("Tokenizer must be trained before saving")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'encoder': self.encoder,
                'bpe_ranks': {' '.join(k): v for k, v in self.bpe_ranks.items()},
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path):
        """Load a tokenizer from a file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.encoder = data['encoder']
        tokenizer.decoder = {int(v): k for k, v in tokenizer.encoder.items()}
        tokenizer.bpe_ranks = {tuple(k.split(' ')): v for k, v in data['bpe_ranks'].items()}
        tokenizer.byte_encoder = bytes_to_unicode()
        tokenizer.byte_decoder = {v: k for k, v in tokenizer.byte_encoder.items()}
        tokenizer.pat = re.compile(r'\s+|\s*([!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~])\s*')
        tokenizer.initialized = True
        
        return tokenizer


def bytes_to_unicode():
    """Returns a dict mapping bytes to unicode strings for text encoding"""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def get_stats(vocab):
    """Count symbol pair frequencies in a vocabulary"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs


def merge(word, first, second):
    """Merge a pair of symbols in a word"""
    word = word.split()
    i = 0
    while i < len(word) - 1:
        if word[i] == first and word[i+1] == second:
            word[i:i+2] = [first + second]
        else:
            i += 1
    return ' '.join(word)
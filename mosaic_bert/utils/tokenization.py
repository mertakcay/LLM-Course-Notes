import os
import json
import regex as re
from typing import List, Dict, Optional, Union

class MosaicBertTokenizer:
    def __init__(
        self,
        vocab_file: str,
        do_lower_case: bool = True,
        max_len: int = None,
        do_basic_tokenize: bool = True,
        never_split: Optional[List[str]] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
    ):
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = {ids: tok for tok, ids in self.vocab.items()}
        self.do_basic_tokenize = do_basic_tokenize
        self.do_lower_case = do_lower_case
        self.max_len = max_len if max_len is not None else int(1e12)
        
        self.never_split = never_split if never_split is not None else []
        self.never_split += [unk_token, sep_token, pad_token, cls_token, mask_token]
        
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.sep_token_id = self.vocab.get(sep_token, 101)
        self.pad_token_id = self.vocab.get(pad_token, 0)
        self.cls_token_id = self.vocab.get(cls_token, 101)
        self.mask_token_id = self.vocab.get(mask_token, 103)
        
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
    
    @staticmethod
    def load_vocab(vocab_file: str) -> Dict[str, int]:
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip('\n')
            vocab[token] = index
        return vocab
    
    def tokenize(self, text: str) -> List[str]:
        if self.do_basic_tokenize:
            tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                if token in self.never_split:
                    tokens.append(token)
                else:
                    tokens.extend(self.wordpiece_tokenizer.tokenize(token))
            return tokens
        else:
            return self.wordpiece_tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.ids_to_tokens.get(id, self.unk_token) for id in ids]
    
    def encode(self, text: str, text_pair: Optional[str] = None, add_special_tokens: bool = True) -> Dict:
        tokens = self.tokenize(text)
        if text_pair is not None:
            tokens_pair = self.tokenize(text_pair)
            return self.prepare_for_model(tokens, pair_tokens=tokens_pair, add_special_tokens=add_special_tokens)
        else:
            return self.prepare_for_model(tokens, add_special_tokens=add_special_tokens)
    
    def prepare_for_model(self, tokens: List[str], pair_tokens: Optional[List[str]] = None, add_special_tokens: bool = True) -> Dict:
        if add_special_tokens:
            if pair_tokens is not None:
                return self.create_token_type_ids_from_sequences(tokens, pair_tokens)
            else:
                return self.build_inputs_with_special_tokens(tokens)
        else:
            return {
                'input_ids': self.convert_tokens_to_ids(tokens),
                'token_type_ids': [0] * len(tokens)
            }
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[str]) -> Dict:
        tokens = [self.cls_token] + token_ids_0 + [self.sep_token]
        input_ids = self.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * len(input_ids)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids
        }
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[str], token_ids_1: List[str]) -> Dict:
        tokens = [self.cls_token] + token_ids_0 + [self.sep_token] + token_ids_1 + [self.sep_token]
        input_ids = self.convert_tokens_to_ids(tokens)
        
        sep_idx = len(token_ids_0) + 2
        token_type_ids = [0] * sep_idx + [1] * (len(tokens) - sep_idx)
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids
        }

class BasicTokenizer:
    def __init__(self, do_lower_case: bool = True, never_split: Optional[List[str]] = None):
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
    
    def tokenize(self, text: str) -> List[str]:
        text = self._clean_text(text)
        
        tokens = []
        for token in text.split():
            if token in self.never_split:
                tokens.append(token)
            else:
                if self.do_lower_case:
                    token = token.lower()
                tokens.extend(self._run_split_on_punc(token))
        
        return tokens
    
    def _clean_text(self, text: str) -> str:
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    
    def _run_split_on_punc(self, text: str) -> List[str]:
        if text in self.never_split:
            return [text]
        
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        
        return ["".join(x) for x in output]
    
    def _is_whitespace(self, char: str) -> bool:
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False
    
    def _is_control(self, char: str) -> bool:
        """Checks whether `char` is a control character."""
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False
    
    def _is_punctuation(self, char: str) -> bool:
        """Checks whether `char` is a punctuation character."""
        cp = ord(char)
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

class WordpieceTokenizer:
    """Runs WordPiece tokenization."""
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_input_chars_per_word: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a piece of text into its word pieces."""
        output_tokens = []
        for token in text.split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            
            is_bad = False
            start = 0
            sub_tokens = []
            
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                
                if cur_substr is None:
                    is_bad = True
                    break
                
                sub_tokens.append(cur_substr)
                start = end
            
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        
        return output_tokens
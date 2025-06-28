import json
from collections import defaultdict, Counter
import config 

class BPETokenizer:
    def __init__(self, vocab_size=30000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab = {}  # token -> id
        self.merges = {} # (token1, token2) -> new_token
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_stats(self, ids):
        counts = defaultdict(int)
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i+1])] += 1
        return counts

    def _merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, corpus_file_path, num_merges=None):
        print(f"Training BPE tokenizer from: {corpus_file_path}")
        
        initial_vocab = set()
        with open(corpus_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                initial_vocab.update(list(line.strip()))
        
        for token in self.special_tokens:
            initial_vocab.add(token)
            
        self.vocab = {char: i for i, char in enumerate(sorted(list(initial_vocab)))}
        self.token_to_id = dict(self.vocab)
        self.id_to_token = {i: char for char, i in self.vocab.items()}
        
        tokenized_corpus = []
        with open(corpus_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split() 
                for word in words:
                    tokenized_corpus.append([self.token_to_id[char] for char in word if char in self.token_to_id])


        if num_merges is None:
            num_merges = self.vocab_size - len(self.vocab)

        print(f"Initial vocabulary size: {len(self.vocab)}")
        print(f"Target number of merges: {num_merges}")

        for i in range(num_merges):
            stats = defaultdict(int)
            for ids in tokenized_corpus:
                current_stats = self._get_stats(ids)
                for pair, count in current_stats.items():
                    stats[pair] += count
            
            if not stats:
                print("No more pairs to merge.")
                break

            best_pair = max(stats, key=stats.get)
            
            if best_pair[0] not in self.id_to_token or best_pair[1] not in self.id_to_token:
                print(f"Skipping merge for pair {best_pair} as one or both tokens are not in id_to_token.")
                del stats[best_pair]
                if not stats:
                    print("No more valid pairs to merge after skipping.")
                    break
                best_pair = max(stats, key=stats.get) 
                if best_pair[0] not in self.id_to_token or best_pair[1] not in self.id_to_token:
                    print(f"Still cannot find valid pair after retry. Stopping merge iteration {i+1}.")
                    break


            new_token_str = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]
            new_token_id = len(self.vocab)
            
            self.vocab[new_token_str] = new_token_id
            self.token_to_id[new_token_str] = new_token_id
            self.id_to_token[new_token_id] = new_token_str
            self.merges[best_pair] = new_token_id
            
            new_tokenized_corpus = []
            for ids in tokenized_corpus:
                new_tokenized_corpus.append(self._merge(ids, best_pair, new_token_id))
            tokenized_corpus = new_tokenized_corpus

            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: Merged {self.id_to_token[best_pair[0]]} + {self.id_to_token[best_pair[1]]} -> {new_token_str} (ID: {new_token_id})")
        
        print(f"Training complete. Final vocabulary size: {len(self.vocab)}")

    def tokenize(self, text):
        if not self.vocab or not self.merges:
            raise ValueError("Tokenizer has not been trained or loaded.")

        words = text.strip().split() 
        all_token_ids = [] 

        for word in words:
            if not word:
                continue
            
            tokens = [char for char in word] 
            
            while True:
                ids_for_word = []
                for token_str in tokens:
                    ids_for_word.append(self.token_to_id.get(token_str, self.token_to_id.get("[UNK]")))

                merged_in_iteration = False 
                temp_new_ids = [] 
                k = 0
                while k < len(ids_for_word):
                    if k < len(ids_for_word) - 1:
                        pair_to_check = (ids_for_word[k], ids_for_word[k+1])
                        if pair_to_check in self.merges:
                            temp_new_ids.append(self.merges[pair_to_check])
                            k += 2 
                            merged_in_iteration = True 
                            continue 
                    temp_new_ids.append(ids_for_word[k])
                    k += 1 
                
                current_word_ids = temp_new_ids 
                tokens = [self.id_to_token[id_] for id_ in current_word_ids]

                if not merged_in_iteration:
                    break 

            for token_str in tokens:
                all_token_ids.append(self.token_to_id.get(token_str, self.token_to_id["[UNK]"]))
        
        return all_token_ids 


    def save_tokenizer(self, save_path):
        tokenizer_data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "merges": {f"{p[0]}_{p[1]}": v for p, v in self.merges.items()} 
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {save_path}")

    @classmethod
    def load_tokenizer(cls, load_path):
        """Loads a tokenizer from saved files."""
        with open(load_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls(vocab_size=tokenizer_data["vocab_size"], special_tokens=tokenizer_data["special_tokens"])
        tokenizer.token_to_id = tokenizer_data["token_to_id"]
        tokenizer.token_to_id = {str(k): v for k, v in tokenizer.token_to_id.items()}

        tokenizer.id_to_token = tokenizer_data["id_to_token"]
        tokenizer.id_to_token = {int(k): v for k, v in tokenizer.id_to_token.items()}
        
        tokenizer.vocab = dict(tokenizer.token_to_id)

        raw_merges = tokenizer_data["merges"]
        tokenizer.merges = {}
        for k_str, v in raw_merges.items():
            p1, p2 = map(int, k_str.split('_'))
            tokenizer.merges[(p1, p2)] = v
            
        print(f"Tokenizer loaded from {load_path}. Vocab size: {len(tokenizer.vocab)}")
        return tokenizer

    def decode(self, token_ids, skip_special_tokens=True):
        # print(f"[TOKENIZER_DEBUG] decode called with token_ids: {token_ids}, skip_special_tokens: {skip_special_tokens}")
        # print(f"[TOKENIZER_DEBUG] self.id_to_token (first 5): {list(self.id_to_token.items())[:5] if isinstance(self.id_to_token, dict) else 'Not a dict'}")
        # print(f"[TOKENIZER_DEBUG] self.special_tokens: {self.special_tokens}")

        if not isinstance(self.id_to_token, dict) or not self.id_to_token:
            # print("[TOKENIZER_DEBUG] id_to_token is not a valid dict or is empty.")
            return "[ERROR:ID_TO_TOKEN_INVALID_OR_EMPTY]"

        output_strings = []
        for token_id_obj in token_ids:
            try:
                current_id = int(token_id_obj) # Ensure token_id is int
            except (ValueError, TypeError) as e:
                # print(f"[TOKENIZER_DEBUG] Could not convert token_id {token_id_obj} to int: {e}. Skipping.")
                output_strings.append("[INVALID_TOKEN_ID_TYPE]")
                continue

            token_string = self.id_to_token.get(current_id)

            if token_string is None:
                # print(f"[TOKENIZER_DEBUG] Token ID {current_id} not found in id_to_token. Using UNK.")
                unk_token_name = "[UNK]"
                unk_id_from_map = self.token_to_id.get(unk_token_name) # Get ID of UNK token
                if unk_id_from_map is not None:
                    token_string = self.id_to_token.get(int(unk_id_from_map), f"[UNK_STR_FOR_ID_{unk_id_from_map}_NOT_FOUND]")
                else:
                    token_string = "[UNK_TOKEN_NAME_NOT_IN_TOKEN_TO_ID]"
            
            if skip_special_tokens and token_string in self.special_tokens:
                # print(f"[TOKENIZER_DEBUG] Skipping special token '{token_string}'")
                continue
            
            output_strings.append(str(token_string)) # Ensure it's a string, though it should be

        # Attempt to join with spaces. This is a heuristic.
        # Joining with empty string to concatenate subword units into words.
        # Spaces between words might need further handling if not part of the token vocabulary itself.
        final_text = "".join(output_strings) 
        # print(f"[TOKENIZER_DEBUG] decode returning (joined with spaces): '{final_text}' (type: {type(final_text)})")
        
        # Ensure a string is always returned
        if final_text is None:
            # print("[TOKENIZER_DEBUG] CRITICAL: final_text is None. Returning empty string instead.")
            return "[CRITICAL_ERROR_DECODE_RETURNED_NONE]"
            
        return final_text.strip() # Remove leading/trailing spaces that might result from join

if __name__ == '__main__':
    corpus_path = config.CORPUS_FILE_PATH
    vocab_size = config.VOCAB_SIZE
    tokenizer_save_path = config.TOKENIZER_PATH

    # --- Test Training ---
    print("\n--- Testing BPE Tokenizer Training ---")
    bpe_tokenizer = BPETokenizer(vocab_size=vocab_size) 
    
    print(f"Starting BPE training with corpus: {corpus_path}")
    print(f"Target vocabulary size: {bpe_tokenizer.vocab_size}")
    print("This may take a while for large corpus and vocabularies...")
    
    bpe_tokenizer.train(corpus_path)
    
    bpe_tokenizer.save_tokenizer(tokenizer_save_path)
    
    # --- Test Loading ---
    print("\n--- Testing BPE Tokenizer Loading ---")
    loaded_tokenizer = BPETokenizer.load_tokenizer(tokenizer_save_path)
    
    # --- Test Tokenization ---
    print("\n--- Testing Tokenization ---")
    test_sentence = "Hayatımın en mutlu anıymış, bilmiyordum." 
    token_ids = loaded_tokenizer.tokenize(test_sentence)
    print(f"Original sentence: {test_sentence}")
    print(f"Token IDs: {token_ids}")
    
    reconstructed_tokens = [loaded_tokenizer.id_to_token.get(id_, "[UNK]") for id_ in token_ids]
    print(f"Reconstructed tokens: {reconstructed_tokens}")

    # Example with special tokens
    print("\n--- Testing Tokenization with Special Tokens ---")
    text_with_special_tokens = "[CLS] Füsun'un gözyaşları [SEP]" 
    token_ids_special = loaded_tokenizer.tokenize(text_with_special_tokens)
    print(f"Original: {text_with_special_tokens}")
    print(f"Token IDs: {token_ids_special}")
    reconstructed_special = [loaded_tokenizer.id_to_token.get(id_, "[UNK]") for id_ in token_ids_special]
    print(f"Reconstructed: {reconstructed_special}")

    test_unknown_word = "Normaleşştiremediklerimizdenmisiniz" 
    token_ids_unknown = loaded_tokenizer.tokenize(test_unknown_word)
    print(f"\nOriginal unknown word: {test_unknown_word}")
    print(f"Token IDs: {token_ids_unknown}")
    reconstructed_unknown = [loaded_tokenizer.id_to_token.get(id_, "[UNK]") for id_ in token_ids_unknown]
    print(f"Reconstructed unknown: {reconstructed_unknown}")
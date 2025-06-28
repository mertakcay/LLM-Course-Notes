import torch
import torch.nn.functional as F
from bert_model import BertConfig, BertForMaskedLM
from tokenizer import BPETokenizer
import config as global_model_config # Assuming your config.py is named config.py

def generate_text(prompt_with_masks: str, model: BertForMaskedLM, tokenizer: BPETokenizer, temperature: float = 0.85, top_k: int = 50, device: str = "cpu") -> str:
    """
    Fills in [MASK] tokens in a given prompt using the BERT model.
    """
    model.eval() # Set the model to evaluation mode
    model.to(device)

    cls_token_str = global_model_config.SPECIAL_TOKENS[2]  # Should be "[CLS]"
    mask_token_str = global_model_config.SPECIAL_TOKENS[4] # Should be "[MASK]"
    unk_token_str = global_model_config.SPECIAL_TOKENS[1]  # Should be "[UNK]"

    try:
        cls_token_id = tokenizer.token_to_id[cls_token_str]
        mask_token_id = tokenizer.token_to_id[mask_token_str]
        unk_token_id = tokenizer.token_to_id[unk_token_str]
    except KeyError as e:
        raise ValueError(f"Essential special token {e} not found in tokenizer vocabulary. Check config.py and tokenizer file.") from e

    # Construct current_token_ids by splitting the prompt by mask_token_str
    # and inserting mask_token_id explicitly.
    current_token_ids = [cls_token_id]
    segments = prompt_with_masks.split(mask_token_str)
    for i, segment_text in enumerate(segments):
        print('segment text', segment_text)
        stripped_segment = segment_text.strip()
        if stripped_segment:  # Tokenize non-empty, stripped segments
            current_token_ids.extend(tokenizer.tokenize(stripped_segment))
        
        if i < len(segments) - 1:  # Add mask_token_id between segments
            print('latest', mask_token_id)
            current_token_ids.append(mask_token_id)
    print(current_token_ids)
    
    # Iteratively fill masks
    num_initial_masks = sum(1 for tid in current_token_ids if tid == mask_token_id)
    max_iterations = num_initial_masks + 5 # Allow a few extra iterations as a buffer

    for iteration_count in range(max_iterations):
        mask_indices = [i for i, token_id in enumerate(current_token_ids) if token_id == mask_token_id]

        if not mask_indices:
            break  # No more masks to fill

        # Predict for the first mask found (left-to-right)
        current_mask_idx_in_sequence = mask_indices[0]

        input_tensor = torch.tensor([current_token_ids], dtype=torch.long).to(device)
        attention_mask_tensor = torch.ones_like(input_tensor).to(device)
        token_type_ids_tensor = torch.zeros_like(input_tensor).to(device) # Single sequence

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_type_ids_tensor)
            
            prediction_scores = outputs[0] if isinstance(outputs, tuple) else outputs

            masked_token_logits = prediction_scores[0, current_mask_idx_in_sequence, :]

            if temperature > 0 and temperature != 1.0: # Avoid division by zero if temp is 0
                masked_token_logits = masked_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0 and top_k < masked_token_logits.size(-1):
                top_k_actual = min(top_k, masked_token_logits.size(-1)) # Ensure top_k is not larger than vocab size
                if top_k_actual > 0: # Ensure top_k is positive
                    top_k_values, top_k_indices = torch.topk(masked_token_logits, top_k_actual)
                    filter_values = torch.full_like(masked_token_logits, -float('Inf'))
                    masked_token_logits = filter_values.scatter_(-1, top_k_indices, top_k_values)

            probabilities = F.softmax(masked_token_logits, dim=-1)
            predicted_token_id = torch.multinomial(probabilities, num_samples=1).item()

            current_token_ids[current_mask_idx_in_sequence] = predicted_token_id
    else: # This else clause executes if the loop completes fully (i.e., max_iterations reached)
        if any(tid == mask_token_id for tid in current_token_ids):
            print(f"Warning: Max iterations ({max_iterations}) reached, but some '[MASK]' tokens may remain.")

    start_index = 0
    if current_token_ids and current_token_ids[0] == cls_token_id:
        start_index = 1
    
    tokens_to_decode = current_token_ids[start_index:]
    
    if not tokens_to_decode:
        return ""
        
    generated_text = tokenizer.decode(tokens_to_decode) # Default skip_special_tokens=True
    return generated_text

def main():
    device = torch.device(global_model_config.DEVICE if torch.cuda.is_available() and global_model_config.DEVICE == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer from {global_model_config.TOKENIZER_PATH}...")
    try:
        tokenizer = BPETokenizer.load_tokenizer(global_model_config.TOKENIZER_PATH)
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {global_model_config.TOKENIZER_PATH}.")
        print("Please ensure the tokenizer is trained and saved correctly.")
        return
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    actual_vocab_size = len(tokenizer.token_to_id)
    pad_token_str = global_model_config.SPECIAL_TOKENS[0] # "[PAD]"
    try:
        pad_token_id = tokenizer.token_to_id[pad_token_str]
    except KeyError:
        print(f"Warning: PAD token '{pad_token_str}' not found in tokenizer. Using default 0. This might be problematic for the model if it expects a specific PAD ID.")
        pad_token_id = 0 # Default to 0 if not found, though this should align with model training

    # --- Initialize Model ---
    print("Initializing BERT model...")
    config = BertConfig(
        vocab_size=actual_vocab_size, # Use actual vocab size from tokenizer
        hidden_size=global_model_config.HIDDEN_SIZE,
        num_hidden_layers=global_model_config.NUM_HIDDEN_LAYERS,
        num_attention_heads=global_model_config.NUM_ATTENTION_HEADS,
        intermediate_size=global_model_config.INTERMEDIATE_SIZE,
        hidden_act=global_model_config.HIDDEN_ACT,
        hidden_dropout_prob=0.0, # Set to 0 for generation/inference
        attention_probs_dropout_prob=0.0, # Set to 0 for generation/inference
        max_position_embeddings=global_model_config.MAX_POSITION_EMBEDDINGS,
        type_vocab_size=global_model_config.TYPE_VOCAB_SIZE,
        initializer_range=global_model_config.INITIALIZER_RANGE,
        pad_token_id=pad_token_id, # Use actual pad_token_id from tokenizer
    )
    model = BertForMaskedLM(config=config)
    
    # --- Load Pre-trained Model Weights ---
    model_path = "/Users/mertakcay/Documents/BERTs/vanilla_bert/bert_mlm_final.pt"
    print(f"Loading pre-trained model weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print("Please ensure the model is trained and saved correctly.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.to(device)
    model.eval() # Ensure model is in evaluation mode

    # --- Fill Masks (MLM-style Generation) ---
    mask_token_for_prompt = global_model_config.SPECIAL_TOKENS[4] # "[MASK]"

    prompt1 = f"The capital of France is {mask_token_for_prompt}."
    print(f"\nFilling masks for prompt: '{prompt1}' (default parameters)")
    filled_text1 = generate_text(prompt1, model, tokenizer, device=device)
    print(f"Filled text: {filled_text1}")

    prompt2 = f"BERT stands for Bidirectional {mask_token_for_prompt} Representations from {mask_token_for_prompt}."
    print(f"\nFilling masks for prompt: '{prompt2}' (temp=0.7, top_k=30)")
    filled_text2 = generate_text(prompt2, model, tokenizer, temperature=0.7, top_k=30, device=device)
    print(f"Filled text: {filled_text2}")

    prompt3 = f"Mert Akçay {mask_token_for_prompt} üniversite {mask_token_for_prompt}."
    print(f"\nFilling masks for prompt: '{prompt3}' (temp=0.0, top_k=10)")
    filled_text3 = generate_text(prompt3, model, tokenizer, temperature=0.0, top_k=10, device=device)
    print(f"Filled text: {filled_text3}")

    prompt4 = f"This model can predict {mask_token_for_prompt} words in a sentence."
    print(f"\nFilling masks for prompt: '{prompt4}' (temp=1.0, top_k=10)")
    filled_text4 = generate_text(prompt4, model, tokenizer, temperature=1.0, top_k=10, device=device)
    print(f"Filled text: {filled_text4}")

if __name__ == "__main__":
    main()
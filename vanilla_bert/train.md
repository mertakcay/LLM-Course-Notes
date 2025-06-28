
## Prerequisites

*   Python 3.7+
*   PyTorch (install via `pip install torch torchvision torchaudio`)
*   A plain text corpus file (`.txt`)

## Step-by-Step Training Process

### 1. Prepare Your Corpus

*   Create a plain text file named `corpus.txt` in the `/Users/mertakcay/Documents/BERTs/vanilla_bert/` directory.
*   This file should contain raw text, with one sentence or document per line (though the BPE tokenizer and dataset script can handle various formats, line-separated text is common).
*   For initial testing, a small file will suffice. For actual pre-training, a large and diverse corpus is crucial.

    Example `corpus.txt`:
    ```
    This is the first sentence.
    BERT is a powerful language model.
    Masked language modeling helps BERT understand context.
    ```

### 2. Configure Parameters (`config.py`)

*   The `config.py` file stores shared parameters for your project, such as file paths, tokenizer settings, model hyperparameters, and training configurations.
*   Review and adjust the parameters in `/Users/mertakcay/Documents/BERTs/vanilla_bert/config.py` as needed. Key parameters include:
    *   `CORPUS_FILE_PATH`: Path to your corpus.
    *   `VOCAB_SIZE`: Desired vocabulary size for the BPE tokenizer.
    *   `MAX_POSITION_EMBEDDINGS`: Maximum sequence length for the model.
    *   Model dimensions (`HIDDEN_SIZE`, `NUM_HIDDEN_LAYERS`, etc.).
    *   Training settings (`BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`).

### 3. Train the BPE Tokenizer (`tokenizer.py`)

The BPE tokenizer learns to segment words into subword units based on the frequency of byte pairs in your corpus.

*   **Implementation**: The `tokenizer.py` script contains the `BPETokenizer` class. The current version in your workspace has a simplified training loop. For robust BPE, the training logic within `BPETokenizer.train()` would need to be fully implemented to iteratively count pairs and merge them based on your `corpus.txt`.
*   **To Run Training (Conceptual for the provided `tokenizer.py`):**
    The `if __name__ == '__main__':` block in `tokenizer.py` demonstrates how to instantiate and train the tokenizer.
    ```bash
    cd /Users/mertakcay/Documents/BERTs/vanilla_bert/
    python tokenizer.py
    ```
*   **Output**: This process will (or should, with a full BPE implementation) generate a `bpe_tokenizer.json` file. This file stores the learned vocabulary, merges (if implemented), and special token mappings. The provided `tokenizer.py` will save a vocabulary based on initial characters, special tokens, and dummy tokens if the full BPE merge logic isn't complete.

    *Note: The `train` method in the provided `/Users/mertakcay/Documents/BERTs/vanilla_bert/bpe_tokenizer.py` appears to be a placeholder or simplified version. A full BPE training implementation is more complex and involves iterative merging based on the corpus statistics.*

### 4. Prepare the MLM Dataset (`dataset.py`)

The `dataset.py` script defines a PyTorch `Dataset` class (`MLMDataset`) that:
*   Loads the raw text from `corpus.txt`.
*   Uses the trained `BPETokenizer` (loaded from `bpe_tokenizer.json`) to convert text into token IDs.
*   Applies the MLM masking strategy:
    *   Adds `[CLS]` and `[SEP]` tokens.
    *   Pads sequences to `max_seq_length`.
    *   Randomly masks 15% of the input tokens.
        *   80% of masked tokens become `[MASK]`.
        *   10% become a random token.
        *   10% remain unchanged.
    *   Generates corresponding `labels` where only masked positions have the true token ID, and others are set to an ignore index (e.g., -100).

This script is primarily used by `train_bert.py` to feed data to the model.

### 5. Define the BERT Model Architecture (`bert_model.py`)

The `bert_model.py` script contains the PyTorch implementation of the BERT model:
*   `BertConfig`: A class to hold model hyperparameters.
*   `BertEmbeddings`: Handles token, position, and segment embeddings.
*   `BertSelfAttention`, `BertAttention`, `BertIntermediate`, `BertOutput`, `BertLayer`: Components of the Transformer encoder.
*   `BertEncoder`: Stacks multiple `BertLayer` instances.
*   `BertLMPredictionHead`: The head used for the MLM task, predicting original tokens from final hidden states.
*   `BertForMaskedLM`: The main model class that combines all components and includes the MLM loss calculation.

This script defines the neural network structure.

### 6. Run the Training Pipeline (`train_bert.py`)

The `train_bert.py` script orchestrates the entire training process:
1.  **Loads Configuration**: Reads settings from `config.py`.
2.  **Loads Tokenizer**: Loads the trained BPE tokenizer from `bpe_tokenizer.json`. If not found, it attempts to train and save a dummy one using `tokenizer.py`.
3.  **Creates Dataset & DataLoader**: Initializes `MLMDataset` and `DataLoader` for batching and shuffling.
4.  **Initializes Model**: Creates an instance of `BertForMaskedLM` using `BertConfig`.
5.  **Sets up Optimizer & Scheduler**: Uses AdamW optimizer and a learning rate scheduler (e.g., linear warmup and decay).
6.  **Training Loop**:
    *   Iterates for `NUM_EPOCHS`.
    *   For each batch:
        *   Moves data to the specified `DEVICE` (GPU or CPU).
        *   Performs a forward pass through the model.
        *   Calculates the MLM loss (CrossEntropyLoss on masked tokens).
        *   Performs a backward pass and updates model weights.
    *   Logs training progress (loss, learning rate).
7.  **Saves Model**: After training, saves the model's state dictionary to `bert_mlm_final.pt` (or as specified in `config.MODEL_SAVE_PATH`).

*   **To Run Training:**
    ```bash
    cd /Users/mertakcay/Documents/BERTs/vanilla_bert/
    python train_bert.py
    ```

### 7. Evaluation (Optional)

After training, you can evaluate your model's performance on a held-out validation set.
*   **Metrics**: Perplexity or accuracy on predicting masked tokens.
*   **Process**:
    1.  Prepare a separate validation text file.
    2.  Modify `train_bert.py` or create a new script (`evaluate.py`) to:
        *   Load the trained model (`bert_mlm_final.pt`).
        *   Set the model to evaluation mode (`model.eval()`).
        *   Process the validation data through the `MLMDataset` (ensuring masking is applied consistently or adapted for evaluation).
        *   Calculate loss and/or accuracy without performing gradient updates.

## File Overview

*   `config.py`: Central configuration for paths, tokenizer, model, and training hyperparameters.
*   `tokenizer.py`: Implements the `BPETokenizer`. Responsible for training on a corpus to learn subword units and for tokenizing raw text into IDs.
*   `dataset.py`: Defines `MLMDataset` for PyTorch. Handles loading text, tokenizing using the trained `BPETokenizer`, and applying the MLM masking strategy.
*   `bert_model.py`: Contains the PyTorch `nn.Module` definitions for the BERT architecture, including embeddings, Transformer encoder layers, and the MLM prediction head.
*   `train_bert.py`: The main script to run the training. It integrates all other components: loads data, initializes the model, and executes the training loop.

---

This guide should help you navigate the process of training your vanilla BERT model. Remember that training large language models from scratch requires significant data and computational resources. The provided setup is a starting point for understanding and experimenting with the core concepts.
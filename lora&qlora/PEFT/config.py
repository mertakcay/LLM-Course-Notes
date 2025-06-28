import torch
import torch

class Config:
    # You can read parameters on comments
    MODEL_NAME = "gpt2"
    
    LORA_R = 16                # Rank of the LoRA update matrices
    LORA_ALPHA = 16           # Scaling factor for LoRA updates
    LORA_DROPOUT = 0.05       # Dropout probability for LoRA layers
    
    # Training hyperparameters
    BATCH_SIZE = 4            # Number of samples processed in each training batch
    GRADIENT_ACCUMULATION_STEPS = 4  # Number of forward passes before updating weights
    LEARNING_RATE = 2e-4      # Step size for gradient descent
    NUM_EPOCHS = 40      # Number of complete passes through the dataset
    MAX_SEQ_LENGTH = 512      # Maximum length of input sequences
    
    # 4-bit quantization parameters for QLoRA
    QUANT_4BIT = False        # Whether to use 4-bit quantization
    BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16  # Data type for computations in 4-bit mode
    BNB_4BIT_QUANT_TYPE = "nf4"  # Quantization type (normal float 4)
    BNB_4BIT_USE_DOUBLE_QUANT = True  # Whether to use nested quantization
    
    # Dataset and output configuration
    DATASET_NAME = "Helsinki-NLP/opus-100"
    DATASET_CONFIG_NAME = "en-tr"  # HuggingFace dataset to use
    OUTPUT_DIR = "./results"   # Directory to save model checkpoints and logs
    
    # Training monitoring parameters
    LOG_STEPS = 10            # How often to log training metrics
    SAVE_STEPS = 10000         # How often to save model checkpoints
    
    # Hardware configuration
    # Training technique selection
    TRAINING_TECHNIQUE = "lora" # Options: "lora", "qlora", "prefix_tuning", "soft_prompting"

    # Prefix Tuning and Soft Prompting parameters
    NUM_VIRTUAL_TOKENS = 10
    PREFIX_PROJECTION = True

    # Mixed precision training
    FP16 = False
    BF16 = False

    # Hardware configuration
    DEVICE = "mps"  # Use GPU if available, else CPU

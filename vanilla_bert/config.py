CORPUS_FILE_PATH = "/Users/mertakcay/Documents/BERTs/vanilla_bert/masumiyet_muzesi.txt"
TOKENIZER_PATH = "/Users/mertakcay/Documents/BERTs/vanilla_bert/bpe_tokenizer_masumiyet.json"
MODEL_SAVE_PATH = "/Users/mertakcay/Documents/BERTs/vanilla_bert/bert_mlm_final.pt"

# Config https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertConfig
VOCAB_SIZE = 30522
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
MASK_TOKEN_ID = 4

HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 12
INTERMEDIATE_SIZE = 3072
HIDDEN_ACT = "gelu"
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2
INITIALIZER_RANGE = 0.02
LAYER_NORM_EPS = 1e-12
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROB = 0.1
MASK_PROB = 0.15

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
WARMUP_STEPS = 100
DEVICE = "mps"
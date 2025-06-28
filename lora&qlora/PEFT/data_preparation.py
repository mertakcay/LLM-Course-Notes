from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from config import Config


def load_and_prepare_dataset(
    model_name: str = Config.MODEL_NAME,
    max_seq_length: int = Config.MAX_SEQ_LENGTH,
    config_name: str = None
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset(Config.DATASET_NAME, config_name) if config_name else load_dataset(Config.DATASET_NAME)

    def tokenize_function(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["tr"] for ex in examples["translation"]]

        model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_seq_length, truncation=True, padding="max_length")

        labels_ids = labels["input_ids"]
        labels_ids = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label_seq]
            for label_seq in labels_ids
        ]
        model_inputs["labels"] = labels_ids
        return model_inputs

    dataset = DatasetDict({
        "train": dataset["train"].select(range(1000)) if "train" in dataset else Dataset.from_dict({}),
        "validation": dataset.get("validation", Dataset.from_dict({})),
        "test": dataset.get("test", Dataset.from_dict({}))
    })

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["translation"])
    return tokenized_dataset, tokenizer

if __name__ == "__main__":
    tokenized_dataset, tokenizer = load_and_prepare_dataset()
    print("Dataset loaded and tokenized")
    print(tokenized_dataset)
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Example from dataset:", tokenized_dataset["train"][0])

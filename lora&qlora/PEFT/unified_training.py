import torch
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_scheduler, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import (
    get_peft_model, prepare_model_for_kbit_training, TaskType,
    LoraConfig, PrefixTuningConfig, PromptTuningConfig
)
from config import Config
from data_preparation import load_and_prepare_dataset
import evaluate


def get_tokenized_dataset_and_tokenizer():
    return load_and_prepare_dataset(
        model_name=Config.MODEL_NAME,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        config_name=Config.DATASET_CONFIG_NAME
    )


def load_base_model(tokenizer):
    if Config.QUANT_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=Config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=Config.BNB_4BIT_USE_DOUBLE_QUANT,
        )
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            device_map="auto" if Config.TRAINING_TECHNIQUE in ["lora", "qlora"] else None,
            trust_remote_code=True
        )

    if Config.TRAINING_TECHNIQUE in ["prefix_tuning", "soft_prompting"]:
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def get_peft_config():
    if Config.TRAINING_TECHNIQUE in ["lora", "qlora"]:
        return LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    elif Config.TRAINING_TECHNIQUE == "prefix_tuning":
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=Config.NUM_VIRTUAL_TOKENS,
            prefix_projection=Config.PREFIX_PROJECTION,
        )
    elif Config.TRAINING_TECHNIQUE == "soft_prompting":
        return PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=Config.NUM_VIRTUAL_TOKENS,
        )
    else:
        raise ValueError("Unsupported TRAINING_TECHNIQUE")


def train_loop(model, train_dataloader, eval_dataloader, tokenizer):
    device = Config.DEVICE
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    num_training_steps = Config.NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    bleu_metric = evaluate.load("bleu")

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n[Epoch {epoch}] Training Started")
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(train_loss))
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}")

        eval_loss, bleu = evaluate_loop(model, eval_dataloader, tokenizer, bleu_metric)
        print(f"[Epoch {epoch}] Eval Loss: {eval_loss:.4f}, Perplexity: {torch.exp(torch.tensor(eval_loss)):.2f}, BLEU: {bleu:.4f}")


def evaluate_loop(model, eval_dataloader, tokenizer, bleu_metric):
    device = Config.DEVICE
    model.eval()
    total_loss = 0
    decoded_preds, decoded_labels = [], []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, -1)
            labels = batch["labels"]
            preds_text = tokenizer.batch_decode(preds.cpu().numpy(), skip_special_tokens=True)
            labels_np = np.where(labels.cpu().numpy() != -100, labels.cpu().numpy(), tokenizer.pad_token_id)
            labels_text = tokenizer.batch_decode(labels_np, skip_special_tokens=True)

            decoded_preds.extend([p.strip() for p in preds_text])
            decoded_labels.extend([[l.strip()] for l in labels_text])

    bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)['bleu']
    return total_loss / len(eval_dataloader), bleu_score


def train_model():
    tokenized_dataset, tokenizer = get_tokenized_dataset_and_tokenizer()
    model = load_base_model(tokenizer)

    if Config.TRAINING_TECHNIQUE in ["lora", "qlora"]:
        model = prepare_model_for_kbit_training(model)

    peft_config = get_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=Config.BATCH_SIZE,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["validation"],
        batch_size=Config.BATCH_SIZE,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    train_loop(model, train_dataloader, eval_dataloader, tokenizer)
    model.save_pretrained(Config.OUTPUT_DIR)


if __name__ == "__main__":
    train_model()

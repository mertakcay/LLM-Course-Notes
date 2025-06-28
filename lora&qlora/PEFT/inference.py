import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import Config


def generate_text(
    prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 50
) -> str:

    inputs = tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def load_model_for_inference() -> tuple[PeftModel, AutoTokenizer]:
    model_name = Config.MODEL_NAME
    peft_model_path = Config.OUTPUT_DIR

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if getattr(Config, "BF16", False) else torch.float16
    if Config.DEVICE == "cpu":
        dtype = torch.float32  

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=Config.DEVICE,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.eval()
    model.to(Config.DEVICE)

    return model, tokenizer


def interactive_generation(model: PeftModel, tokenizer: AutoTokenizer):
    print(f"\nModel ready on device: {Config.DEVICE}. Enter prompt (type 'exit' to quit):\n")
    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() == "exit":
                break
            generated = generate_text(prompt, model, tokenizer)
            print("\nGenerated:\n", generated)
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"[ERROR] {e}")
            continue


if __name__ == "__main__":
    print(f"Loading model from {Config.OUTPUT_DIR}...")
    model, tokenizer = load_model_for_inference()
    print("Model loaded successfully.")
    interactive_generation(model, tokenizer)

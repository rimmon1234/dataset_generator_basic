import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_ID = "Qwen/Qwen3-1.7B"

def load_model_and_tokenizer(model_id: str = MODEL_ID):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "NVIDIA CUDA GPU is required for 4-bit bitsandbytes quantization."
        )

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure pad token exists and padding side is set to left for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=compute_dtype
    )

    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    print(f"Loading {MODEL_ID} on {torch.cuda.get_device_name(0)}...")
    model, tokenizer = load_model_and_tokenizer()
    print("Model loaded successfully!")
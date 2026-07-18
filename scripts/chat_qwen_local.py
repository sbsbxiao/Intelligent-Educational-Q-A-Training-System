from __future__ import annotations

from pathlib import Path

from config import settings

ROOT_DIR = Path(__file__).resolve().parents[1]


def get_model_path() -> Path:
    if not settings.local_text_generation_path:
        raise ValueError("LOCAL_TEXT_GENERATION_PATH is not configured in .env")

    model_path = Path(settings.local_text_generation_path)
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path
    return model_path


def load_model(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading model: {model_path}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个简洁、准确的中文助手。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=settings.local_text_generation_max_new_tokens,
        do_sample=False,
    )
    output_ids = output_ids[:, model_inputs.input_ids.shape[-1]:]
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tokenizer, model = load_model(model_path)
    print("模型已加载。输入问题开始对话，输入 exit 退出。")

    while True:
        try:
            prompt = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            return

        if prompt.lower() in {"exit", "quit", "q"}:
            print("退出。")
            return
        if not prompt:
            continue

        answer = generate(tokenizer, model, prompt)
        print(f"助手: {answer}")


if __name__ == "__main__":
    main()

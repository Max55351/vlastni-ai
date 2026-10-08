"""
Chat s vlastnim doladenym asistentem (faze 2).

    python assistant.py
    python assistant.py --base-only     # zaklad bez tveho adapteru, pro porovnani
"""
import argparse
from pathlib import Path

import torch

OUT = Path(__file__).parent / "out" / "lora"

SYSTEM = "Jsi osobni asistent. Odpovidas strucne, vecne a v jazyce, kterym se te uzivatel pta."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import finetune

    base = args.model or finetune.pick_model()
    use_lora = not args.base_only and (OUT / "adapter_config.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(str(OUT) if use_lora else base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )
    if use_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, OUT)
        print(f"nacten tvuj adapter z {OUT}")
    else:
        print("bezi cisty zaklad bez tveho doladeni")
    model.eval()

    history = [{"role": "system", "content": SYSTEM}]
    print("Pis, Ctrl+C konci. /reset smaze historii.\n")
    while True:
        try:
            user = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not user:
            continue
        if user == "/reset":
            history = history[:1]
            print("historie smazana\n")
            continue

        history.append({"role": "user", "content": user})
        ids = tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.max_new, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        print(reply + "\n")


if __name__ == "__main__":
    main()

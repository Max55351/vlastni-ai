"""
FAZE 2 - vlastni AI asistent: QLoRA fine-tune otevreneho modelu na tvych datech.

Na rozdil od faze 1 (model od nuly) tady stavis na otevrenem zakladu, takze
vysledek umi opravdu konverzovat. Vahy adapteru jsou tve a vsechno bezi lokalne.

Data: data/chat.jsonl, jeden priklad na radek:
    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Pouziti:
    python finetune.py --check                 # jen ukaz, co by se delo
    python finetune.py                         # autodetekce modelu podle VRAM
    python finetune.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out" / "lora"

# volba zakladniho modelu podle toho, kolik se veji do VRAM ve 4 bitech
BY_VRAM = [
    (23, "Qwen/Qwen2.5-14B-Instruct"),
    (15, "Qwen/Qwen2.5-7B-Instruct"),
    (7.5, "Qwen/Qwen2.5-7B-Instruct"),
    (5.5, "Qwen/Qwen2.5-3B-Instruct"),
    (0, "Qwen/Qwen2.5-1.5B-Instruct"),
]


def pick_model():
    if not torch.cuda.is_available():
        return BY_VRAM[-1][1]
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return next(m for th, m in BY_VRAM if gb >= th)


def load_dataset(tokenizer, max_len):
    path = DATA / "chat.jsonl"
    if not path.exists():
        raise SystemExit(
            f"Chybi {path}.\nFormat - jeden JSON na radek:\n"
            '{"messages":[{"role":"user","content":"ahoj"},'
            '{"role":"assistant","content":"ahoj, co potrebujes?"}]}'
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        enc = tokenizer(text, truncation=True, max_length=max_len)
        enc["labels"] = enc["input_ids"].copy()
        rows.append(enc)
    print(f"nacteno {len(rows)} trenovacich prikladu")
    if len(rows) < 50:
        print("  ! varovani: pod ~50 prikladu se model nic poradne nenauci")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    base = args.model or pick_model()
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    print(f"GPU {gb:.1f} GB -> zakladni model: {base}")
    if args.check:
        print("--check: nic se nestahuje ani netrenuje")
        return

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_dataset(tokenizer, args.max_len)

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    Trainer(
        model=model,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        args=TrainingArguments(
            output_dir=str(OUT),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=torch.cuda.is_available(),
            logging_steps=5,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            report_to=[],
        ),
    ).train()

    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"\nhotovo, adapter v {OUT}\nZkus to: python assistant.py")


if __name__ == "__main__":
    main()

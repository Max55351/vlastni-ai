"""
Trenink vlastniho modelu.

Pouziti:
    python train.py                       # autodetekce podle GPU
    python train.py --preset small
    python train.py --resume              # pokracuj z posledniho checkpointu

Trenink lze kdykoliv prerusit pres Ctrl+C - checkpoint se ulozi.
"""
import argparse
import math
import os
import time
from pathlib import Path

# Musi byt nastaveno drive, nez se nacte torch - snizuje fragmentaci pameti
# grafiky, coz je na 6GB karte rozdil mezi "bezi" a "dosla pamet".
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

import config
from model import GPT
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent
OUT = ROOT / "out"

OOM_RETRIES = 20  # pri kratkodobem vypadku pameti grafiky se krok zopakuje


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(step, base_lr, warmup, max_steps, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step > max_steps:
        return base_lr * min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, data, cfg, device, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, cfg["model"].block_size, cfg["batch_size"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--compile", action="store_true", help="torch.compile - rychlejsi, ale dlouhy start")
    ap.add_argument("--vykon", type=float, default=1.0, metavar="0-1",
                    help="jakou cast casu smi trenink zabirat grafiku "
                         "(0.5 = pulka vykonu, zbytek zbyde na hry)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="prepis velikost davky (mensi = min pameti grafiky)")
    args = ap.parse_args()

    if not 0.05 <= args.vykon <= 1.0:
        raise SystemExit("--vykon musi byt mezi 0.05 a 1.0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name, cfg = config.get(args.preset)
    mcfg = cfg["model"]
    max_steps = args.max_steps or cfg["max_steps"]

    # Mensi davka = mensi zabrana pamet grafiky. Aby to nezmenilo vysledek
    # treninku, dorovna se to vetsim poctem nasobenych pruchodu - pocet
    # tokenu na krok zustane stejny, jen se spocitaji po mensich castech.
    if args.batch_size:
        old_b, old_a = cfg["batch_size"], cfg["grad_accum"]
        cfg = dict(cfg, batch_size=args.batch_size,
                   grad_accum=max(1, round(old_b * old_a / args.batch_size)))
        print(f"davka {old_b}x{old_a} -> {args.batch_size}x{cfg['grad_accum']} "
              f"(stejne tokenu na krok, mene pameti grafiky)")

    if not (OUT / "train.bin").exists():
        raise SystemExit("Chybi out/train.bin - spust nejdriv: python prepare.py")

    tok = BPETokenizer.load(OUT / "tokenizer.json")
    mcfg.vocab_size = tok.vocab_size

    dtype = np.uint16 if mcfg.vocab_size < 65536 else np.uint32
    train_data = np.memmap(OUT / "train.bin", dtype=dtype, mode="r")
    val_data = np.memmap(OUT / "val.bin", dtype=dtype, mode="r")

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = GPT(mcfg).to(device)
    print(f"preset {name}: {model.num_params() / 1e6:.1f}M parametru, "
          f"kontext {mcfg.block_size}, slovnik {mcfg.vocab_size}")
    print(f"data: {len(train_data):,} trenovacich tokenu")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # weight decay jen na maticove vahy, ne na normalizace a embeddingy
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, fused=(device == "cuda"),
    )

    start_step, best_val = 0, float("inf")
    ckpt_path = OUT / "model.pt"
    if args.resume:
        # navazuje se od toho, co je dal - model.pt je nejlepsi podle kontrolnich
        # dat, model_last.pt je nejnovejsi, a ten byva o kus napred
        cands = [(torch.load(p, map_location="cpu", weights_only=False)["step"], p)
                 for p in (ckpt_path, OUT / "model_last.pt") if p.exists()]
        if not cands:
            raise SystemExit("neni z ceho navazat - v out/ chybi checkpoint")
        _, newest = max(cands)
        ck = torch.load(newest, map_location=device, weights_only=False)
        print(f"navazuji ze souboru {newest.name}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_step, best_val = ck["step"], ck.get("best_val", float("inf"))
        print(f"navazuji od kroku {start_step} (nejlepsi val loss {best_val:.4f})")

    if args.compile:
        model = torch.compile(model)

    warmup = min(500, max_steps // 20)
    eval_every = min(500, max(20, max_steps // 10))  # i kratky beh se musi zkontrolovat
    accum = cfg["grad_accum"]
    tokens_per_step = cfg["batch_size"] * accum * mcfg.block_size
    print(f"{tokens_per_step:,} tokenu na krok, {max_steps:,} kroku "
          f"= {tokens_per_step * max_steps / 1e9:.2f}B tokenu celkem\n")

    t0 = time.time()
    try:
        for step in range(start_step, max_steps):
            step_t0 = time.time()
            lr = lr_at(step, cfg["lr"], warmup, max_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            # Kdyz si jina aplikace vezme pamet grafiky, krok spadne na OOM.
            # Neni duvod kvuli tomu zahodit cely trenink - uklidime pamet,
            # chvili pockame a krok zopakujeme.
            for attempt in range(OOM_RETRIES):
                try:
                    opt.zero_grad(set_to_none=True)
                    total = 0.0
                    for _ in range(accum):
                        x, y = get_batch(train_data, mcfg.block_size,
                                         cfg["batch_size"], device)
                        with torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=device == "cuda"):
                            _, loss = model(x, y)
                        (loss / accum).backward()
                        total += loss.item() / accum

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    break
                except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    wait = 15 * (attempt + 1)
                    print(f"  ! dosla pamet grafiky (pokus {attempt + 1}/{OOM_RETRIES}), "
                          f"cekam {wait}s a zkousim znovu", flush=True)
                    time.sleep(wait)
            else:
                print("  !! pamet grafiky nedostupna ani po opakovani - ukladam a koncim")
                save(model, opt, step, best_val, mcfg, name, OUT / "model_last.pt")
                raise SystemExit(1)

            # Brzda: po vypoctu se na chvili pusti grafika, aby na ni stihlo
            # bezet i neco jineho. --vykon 0.5 znamena pulku casu pocitat,
            # pulku cekat.
            if args.vykon < 1.0:
                time.sleep((time.time() - step_t0) * (1 / args.vykon - 1))

            if step % 10 == 0:
                el = time.time() - t0
                done = step - start_step + 1
                eta = el / done * (max_steps - step - 1)
                mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
                print(f"krok {step:6d}/{max_steps}  loss {total:.4f}  lr {lr:.2e}  "
                      f"{el / done:.2f}s/krok  VRAM {mem:.1f}GB  "
                      f"zbyva {eta / 3600:.1f}h", flush=True)

            # Prubezna zaloha, aby pad nikdy nestal vic nez par minut prace.
            # step != start_step: hned po navazani se jeste nic nespocitalo,
            # a zapis by prepsal zalohu starsim stavem.
            if step > 0 and step != start_step and step % 100 == 0:
                save(model, opt, step, best_val, mcfg, name, OUT / "model_last.pt")

            if step > 0 and step % eval_every == 0:
                vl = evaluate(model, val_data, cfg, device)
                print(f"  >> val loss {vl:.4f}  (perplexita {math.exp(min(vl, 20)):.1f})")
                if vl < best_val:
                    best_val = vl
                    save(model, opt, step, best_val, mcfg, name, ckpt_path)
                    print(f"  >> ulozeno do {ckpt_path.name}")
    except KeyboardInterrupt:
        print("\npreruseno - ukladam checkpoint")
        save(model, opt, step, best_val, mcfg, name, OUT / "model_last.pt")
        return

    save(model, opt, max_steps, best_val, mcfg, name, OUT / "model_last.pt")
    print(f"\nhotovo za {(time.time() - t0) / 3600:.2f}h. "
          f"Zkus to: python chat.py")


def save(model, opt, step, best_val, mcfg, preset, path):
    sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({
        "model": sd, "optim": opt.state_dict(), "step": step,
        "best_val": best_val, "cfg": mcfg, "preset": preset,
    }, path)


if __name__ == "__main__":
    main()

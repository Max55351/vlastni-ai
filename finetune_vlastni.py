"""
Nauci TVUJ model odpovidat na otazky (instrukcni doladeni).

Zadny cizi model - bere out/model.pt (tvuj model od nuly) a uci ho na dvojicich
otazka -> odpoved z out/qa_train.jsonl, ktere vyrobi make_qa.py z tve Wikipedie.
Chyba se pocita jen z odpovedi: model se ma naucit odpovidat, ne vymyslet otazky.
Vysledek se ulozi do out/model_qa.pt, puvodni model zustane beze zmeny.

    python finetune_vlastni.py
    python finetune_vlastni.py --kroku 3000 --vykon 0.5
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from model import GPT
from tokenizer import BPETokenizer

OUT = Path(__file__).parent / "out"
SABLONA = "Otázka: {q}\nOdpověď:"
KONEC = "<|endoftext|>"
ZKUSEBNI_OTAZKY = ["Kdo byl Napoleon Bonaparte?", "Co je Praha?", "Kdo byl Karel IV.?"]


def nacti(cesta, tok, max_delka):
    """Vrati seznam (vstup, cile). Cile otazky jsou -1 = nepocitat do chyby."""
    konec = tok.special[KONEC]
    vzorky = []
    with open(cesta, encoding="utf-8") as f:
        for radek in f:
            r = json.loads(radek)
            otazka = tok.encode(SABLONA.format(q=r["otazka"]))
            odpoved = tok.encode(" " + r["odpoved"]) + [konec]
            ids = (otazka + odpoved)[: max_delka + 1]
            if len(ids) <= len(otazka):
                continue  # odpoved by se cela usekla
            vstup = ids[:-1]
            # cil na pozici j je ids[j+1]; dokud je to jeste otazka, nepocita se
            cile = [-1 if j < len(otazka) - 1 else ids[j + 1] for j in range(len(vstup))]
            vzorky.append((vstup, cile))
    return vzorky


def davka(vzorky, velikost, pad, device):
    vyber = random.sample(vzorky, velikost)
    delka = max(len(v) for v, _ in vyber)
    x = torch.full((velikost, delka), pad, dtype=torch.long)
    y = torch.full((velikost, delka), -1, dtype=torch.long)
    for i, (v, c) in enumerate(vyber):
        x[i, : len(v)] = torch.tensor(v)
        y[i, : len(c)] = torch.tensor(c)
    return x.to(device), y.to(device)


def lr_v_kroku(krok, zakladni, zahrati, celkem):
    if krok < zahrati:
        return zakladni * (krok + 1) / zahrati
    pokrok = (krok - zahrati) / max(1, celkem - zahrati)
    return zakladni * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, pokrok))))


@torch.no_grad()
def kontrola(model, vzorky, velikost, pad, device, opakovani=20):
    model.eval()
    ztraty = []
    for _ in range(opakovani):
        x, y = davka(vzorky, min(velikost, len(vzorky)), pad, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        ztraty.append(loss.item())
    model.train()
    return sum(ztraty) / len(ztraty)


@torch.no_grad()
def ukazka(model, tok, device):
    konec = tok.special[KONEC]
    for q in ZKUSEBNI_OTAZKY:
        ids = tok.encode(SABLONA.format(q=q))
        out = model.generate(torch.tensor([ids], device=device), 80, temperature=0.5,
                             repetition_penalty=1.15, stop_ids=(konec,))
        text = tok.decode(out[0, len(ids):].tolist()).split(KONEC)[0].strip()
        print(f"     {q} -> {text[:160]}")
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kroku", type=int, default=3000)
    ap.add_argument("--davka", type=int, default=32)
    ap.add_argument("--mikro", type=int, default=8,
                    help="kolik otazek se pocita najednou - cela davka 32 x 256 tokenu se do 6 GB "
                         "grafiky nevejde, Windows ji preleje do RAM a trenink je 15x pomalejsi")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-delka", type=int, default=256)
    ap.add_argument("--vykon", type=float, default=1.0,
                    help="jakou cast casu smi trenink zabirat grafiku (0.5 = pulka)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(OUT / "tokenizer.json")
    ck = torch.load(OUT / "model.pt", map_location=device, weights_only=False)
    model = GPT(ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.train()

    trenovaci = nacti(OUT / "qa_train.jsonl", tok, args.max_delka)
    kontrolni = nacti(OUT / "qa_val.jsonl", tok, args.max_delka)
    pad = tok.special[KONEC]
    print(f"tvuj model: {model.num_params() / 1e6:.1f}M parametru (krok {ck['step']} "
          f"predtreninku) | dvojic otazka-odpoved: {len(trenovaci):,} + {len(kontrolni):,} kontrolnich")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=(device == "cuda"))
    zahrati = min(100, args.kroku // 10)
    mikro_kroku = max(1, args.davka // args.mikro)
    nejlepsi = float("inf")
    print("\npred doladenim:")
    ukazka(model, tok, device)

    t0 = time.time()
    for krok in range(args.kroku):
        kt0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_v_kroku(krok, args.lr, zahrati, args.kroku)
        for pokus in range(10):
            try:
                # davka se pocita po castech a gradienty se sectou - vysledek
                # je stejny jako pro celou davku, ale vejde se do grafiky
                opt.zero_grad(set_to_none=True)
                for _ in range(mikro_kroku):
                    x, y = davka(trenovaci, args.mikro, pad, device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        _, loss = model(x, y)
                    (loss / mikro_kroku).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                break
            except torch.OutOfMemoryError:
                # hra si vzala pamet grafiky - uklidit, chvili pockat, zkusit znovu
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"  ! dosla pamet grafiky, pokus {pokus + 1}/10", flush=True)
                time.sleep(15 * (pokus + 1))
        else:
            raise SystemExit("pamet grafiky trvale nedostupna - koncim")

        if args.vykon < 1.0:
            time.sleep((time.time() - kt0) * (1 / args.vykon - 1))

        if krok % 25 == 0:
            el = time.time() - t0
            print(f"krok {krok:5d}/{args.kroku}  chyba {loss.item():.3f}  "
                  f"{el / (krok + 1):.2f}s/krok  zbyva {el / (krok + 1) * (args.kroku - krok) / 60:.0f} min",
                  flush=True)

        if (krok > 0 and krok % 250 == 0) or krok == args.kroku - 1:
            vl = kontrola(model, kontrolni, args.mikro, pad, device, opakovani=80)
            print(f"  >> kontrolni chyba odpovedi {vl:.3f}", flush=True)
            ukazka(model, tok, device)
            if vl < nejlepsi:
                nejlepsi = vl
                torch.save({"model": model.state_dict(), "cfg": ck["cfg"], "step": krok,
                            "best_val": vl, "preset": "qa", "zaklad_krok": ck["step"]},
                           OUT / "model_qa.pt")
                print("  >> ulozeno do out/model_qa.pt", flush=True)

    print(f"\nhotovo za {(time.time() - t0) / 60:.0f} min, nejlepsi kontrolni chyba {nejlepsi:.3f}")


if __name__ == "__main__":
    main()

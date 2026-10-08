"""
Povidani s vlastnim modelem.

    python chat.py                     # interaktivni rezim
    python chat.py -p "Bylo nebylo"    # jednorazove doplneni
"""
import argparse
from pathlib import Path

import torch

from model import GPT
from tokenizer import BPETokenizer

OUT = Path(__file__).parent / "out"


def load(ckpt_name=None):
    for cand in ([ckpt_name] if ckpt_name else ["model.pt", "model_last.pt"]):
        p = OUT / cand
        if p.exists():
            break
    else:
        raise SystemExit("Zadny natrenovany model v out/ - spust nejdriv: python train.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(p, map_location=device, weights_only=False)
    model = GPT(ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    tok = BPETokenizer.load(OUT / "tokenizer.json")
    print(f"{p.name}: {model.num_params() / 1e6:.1f}M parametru, "
          f"krok {ck['step']}, val loss {ck.get('best_val', float('nan')):.4f}")
    return model, tok, device


KONEC_CLANKU = "<|endoftext|>"


def generate(model, tok, device, prompt, n=200, temp=0.8, top_k=50, top_p=0.95,
             repetition_penalty=1.15, stop_at_end=True):
    ids = tok.encode(prompt) or [0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    konec = tok.special.get(KONEC_CLANKU)
    stop_ids = (konec,) if stop_at_end and konec is not None else ()
    out = model.generate(x, n, temperature=temp, top_k=top_k, top_p=top_p,
                         repetition_penalty=repetition_penalty, stop_ids=stop_ids)
    # bez konce clanku by model plynule preskocil na uplne jine tema
    return tok.decode(out[0].tolist()[len(ids):]).split(KONEC_CLANKU)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt")
    ap.add_argument("-n", "--tokens", type=int, default=200)
    ap.add_argument("-t", "--temperature", type=float, default=0.8)
    ap.add_argument("--ckpt")
    args = ap.parse_args()

    model, tok, device = load(args.ckpt)

    if args.prompt:
        print(args.prompt + generate(model, tok, device, args.prompt,
                                     args.tokens, args.temperature))
        return

    print("Pis prompt, Ctrl+C konci.\n")
    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not prompt.strip():
            continue
        print(generate(model, tok, device, prompt, args.tokens, args.temperature))
        print()


if __name__ == "__main__":
    main()

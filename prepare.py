"""
Priprava dat: vezme vsechny .txt soubory z data/, natrenuje na nich tokenizer
a ulozi tokeny jako binarni soubory pro trenink.

Zpracovava se po kouscich a zapisuje rovnou na disk, takze to zvladne
i korpus vetsi nez operacni pamet.

Pouziti:
    python prepare.py                    # autodetekce presetu podle GPU
    python prepare.py --preset small
    python prepare.py --sample-mb 5      # mensi vzorek na trenink tokenizeru
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import config
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out"

CHUNK = 4 * 1024 * 1024  # kolik znaku zpracovat najednou

_worker_tok = None


def _init_worker(tok_path):
    global _worker_tok
    _worker_tok = BPETokenizer.load(tok_path)


def _encode_chunk(text):
    return _worker_tok.encode(text)


def read_chunks(files, chunk=CHUNK):
    """Cte soubory po kouscich, ale nikdy nerozdeli slovo uprostred."""
    rest = ""
    for path in files:
        with path.open(encoding="utf-8", errors="ignore") as f:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                data = rest + data
                cut = data.rfind("\n")
                if cut <= 0:
                    rest = data
                    continue
                yield data[:cut + 1]
                rest = data[cut + 1:]
        rest += "\n\n"
    if rest.strip():
        yield rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None)
    ap.add_argument("--val-split", type=float, default=0.005)
    ap.add_argument("--sample-mb", type=int, default=10,
                    help="kolik MB textu pouzit na trenink tokenizeru")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    name, cfg = config.get(args.preset)
    vocab_size = cfg["model"].vocab_size
    print(f"preset: {name}  (slovnik {vocab_size} tokenu)")

    files = sorted(DATA.glob("**/*.txt"))
    if not files:
        raise SystemExit(
            f"V {DATA} nejsou zadne .txt soubory.\n"
            "Bud spust python download_wiki.py, nebo tam nasyp vlastni texty."
        )
    total_mb = sum(f.stat().st_size for f in files) / 1024**2
    print(f"{len(files)} souboru, {total_mb:.0f} MB textu")

    OUT.mkdir(exist_ok=True)

    # --- 1. tokenizer se uci na vzorku, na celku by to bylo zbytecne dlouhe ---
    tok_path = OUT / "tokenizer.json"
    print(f"trenuji tokenizer na vzorku {args.sample_mb} MB...")
    sample, got = [], 0
    for chunk in read_chunks(files):
        sample.append(chunk)
        got += len(chunk)
        if got >= args.sample_mb * 1024**2:
            break
    t0 = time.time()
    BPETokenizer().train("".join(sample), vocab_size=vocab_size).save(tok_path)
    del sample
    print(f"  trvalo {time.time() - t0:.0f}s")

    tok = BPETokenizer.load(tok_path)

    # --- 2. tokenizace celeho korpusu, paralelne a rovnou na disk ---
    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32
    tmp = OUT / "all.bin"
    print(f"tokenizuji {total_mb:.0f} MB na {args.workers} jadrech...")

    t0, done_mb, total_tokens = time.time(), 0.0, 0
    with tmp.open("wb") as fout, Pool(args.workers, _init_worker, (tok_path,)) as pool:
        for ids in pool.imap(_encode_chunk, read_chunks(files), chunksize=1):
            np.asarray(ids, dtype=dtype).tofile(fout)
            total_tokens += len(ids)
            done_mb += CHUNK / 1024**2
            el = time.time() - t0
            pct = min(100.0, done_mb / total_mb * 100)
            eta = el / max(pct, 0.1) * (100 - pct)
            print(f"  {pct:5.1f} %  {total_tokens:,} tokenu  "
                  f"zbyva {eta / 60:.0f} min   ", end="\r")
    print()

    # --- 3. rozdeleni na trenovaci a kontrolni cast ---
    arr = np.memmap(tmp, dtype=dtype, mode="r")
    split = int(len(arr) * (1 - args.val_split))
    with (OUT / "train.bin").open("wb") as f:
        for i in range(0, split, 10_000_000):
            arr[i:min(i + 10_000_000, split)].tofile(f)
    arr[split:].tofile(OUT / "val.bin")
    del arr
    tmp.unlink()

    print(f"hotovo: {total_tokens:,} tokenu "
          f"(train {split:,} / val {total_tokens - split:,})")
    print(f"komprese: {total_mb * 1024**2 / total_tokens:.2f} znaku na token")
    print(f"\nDalsi krok: python train.py")


if __name__ == "__main__":
    main()

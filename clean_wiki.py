"""
Prezene uz stazeny wikipedia_*.txt jeste jednou cisticem z download_wiki.py.
Diky tomu neni potreba stahovat gigabajt znovu, kdyz se cistic zlepsi.

    python clean_wiki.py data/wikipedia_cs.txt
"""
import sys
from pathlib import Path

from download_wiki import clean

SEP = "\n\n<|endoftext|>\n\n"


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/wikipedia_cs.txt")
    if not src.exists():
        raise SystemExit(f"neexistuje: {src}")

    before = src.stat().st_size
    dst = src.with_suffix(".clean.txt")

    kept = dropped = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        buf = []
        for line in fin:
            if line.rstrip("\n") == "<|endoftext|>":
                text = clean("".join(buf))
                if len(text) >= 500:
                    fout.write(text + SEP)
                    kept += 1
                else:
                    dropped += 1
                buf = []
                if (kept + dropped) % 20000 == 0:
                    print(f"  {kept + dropped:,} clanku", end="\r")
            else:
                buf.append(line)

    after = dst.stat().st_size
    src.unlink()
    dst.rename(src)
    print(f"\n{kept:,} clanku ({dropped:,} zahozeno jako prilis kratke)")
    print(f"{before / 1024**2:.0f} MB -> {after / 1024**2:.0f} MB "
          f"(oriznuto {100 * (before - after) / before:.1f} % balastu)")


if __name__ == "__main__":
    main()

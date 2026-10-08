"""
Stahne ceskou Wikipedii jako cisty text do data/wikipedia_cs.txt.
Zdroj: wikimedia/wikipedia na HuggingFace (uz zbaveno wiki znacek), licence CC BY-SA.

    python download_wiki.py                 # cela, ~1 GB
    python download_wiki.py --limit 50000   # jen prvnich 50k clanku (test)
    python download_wiki.py --lang en
"""
import argparse
import re
from pathlib import Path

DATA = Path(__file__).parent / "data"

# Sekce na konci clanku, ktere jsou jen seznamy odkazu. V textu z HF uz nejsou
# wiki znacky '==', takze nadpis je proste samostatny radek s timhle slovem -
# hleda se obojí, aby cistic fungoval i na surovy wikitext.
JUNK_WORDS = (r"Reference|Externí odkazy|Související články|Literatura|Odkazy|"
              r"Poznámky|Bibliografie|References|External links|See also|"
              r"Further reading|Bibliography")
JUNK_SECTIONS = re.compile(rf"\n=* *({JUNK_WORDS}) *=*[ \t]*\n.*", re.DOTALL)


def clean(text: str) -> str:
    text = re.sub(r"==+ *(.+?) *=+", r"\1", text)   # nadpisy na holy text
    text = JUNK_SECTIONS.sub("", text)              # utni balast na konci
    text = re.sub(r"\n{3,}", "\n\n", text)          # zbytecne prazdne radky
    text = re.sub(r"[ \t]+\n", "\n", text)          # mezery na konci radku
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="cs")
    ap.add_argument("--limit", type=int, default=None, help="max poctu clanku")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="preskoc pahyly kratsi nez tohle")
    ap.add_argument("--snapshot", default="20231101")
    args = ap.parse_args()

    from datasets import load_dataset

    DATA.mkdir(exist_ok=True)
    out = DATA / f"wikipedia_{args.lang}.txt"

    print(f"stahuji {args.lang}.wikipedia ({args.snapshot})...")
    print("prvni spusteni stahuje ~1 GB, muze to trvat desitky minut")

    ds = load_dataset(
        "wikimedia/wikipedia", f"{args.snapshot}.{args.lang}",
        split="train", streaming=True,
    )

    kept = skipped = chars = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = clean(row["text"])
            if len(text) < args.min_chars:
                skipped += 1
                continue
            f.write(f"{row['title']}\n\n{text}\n\n<|endoftext|>\n\n")
            kept += 1
            chars += len(text)
            if kept % 5000 == 0:
                print(f"  {kept:,} clanku, {chars / 1024**2:.0f} MB", end="\r")
            if args.limit and kept >= args.limit:
                break

    print(f"\n{out}")
    print(f"{kept:,} clanku ({skipped:,} pahylu preskoceno), "
          f"{out.stat().st_size / 1024**2:.0f} MB")
    print("\nDalsi krok: python prepare.py")


if __name__ == "__main__":
    main()

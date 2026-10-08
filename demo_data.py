"""
Vygeneruje umely testovaci korpus do data/demo.txt.
Slouzi jen k overeni, ze cely pipeline funguje - nez nasypes vlastni data.
"""
import random
from pathlib import Path

DATA = Path(__file__).parent / "data"

SUBJ = ["kocka", "pes", "clovek", "stroj", "rybar", "dite", "starec", "mesto", "reka", "vitr"]
VERB = ["videl", "hledal", "ztratil", "nasel", "postavil", "zapomnel", "slysel", "premyslel o"]
OBJ = ["klic", "dopis", "cestu domu", "stare hodiny", "jmeno", "mapu", "svetlo v okne", "pisen"]
PLACE = ["v lese", "na nadrazi", "u more", "za mestem", "v prazdnem dome", "na moste"]
TIME = ["Rano", "V noci", "Kdyz zacalo prset", "Po tech letech", "Nez slunce zapadlo"]


def sentence(rng):
    return (f"{rng.choice(TIME)} {rng.choice(SUBJ)} {rng.choice(VERB)} "
            f"{rng.choice(OBJ)} {rng.choice(PLACE)}.")


def main(n_docs=4000):
    rng = random.Random(42)
    DATA.mkdir(exist_ok=True)
    docs = []
    for _ in range(n_docs):
        docs.append(" ".join(sentence(rng) for _ in range(rng.randint(5, 15))))
    text = "\n\n".join(docs)
    out = DATA / "demo.txt"
    out.write_text(text, encoding="utf-8")
    print(f"{out}: {len(text) / 1024**2:.2f} MB, {n_docs} odstavcu")
    print("Tohle je JEN test pipeline. Pro skutecny model nahrad vlastnimi texty.")


if __name__ == "__main__":
    main()

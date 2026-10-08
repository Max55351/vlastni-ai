"""
Vyrobi z tve Wikipedie dvojice otazka -> odpoved pro doladeni tveho modelu.

Otazka vznikne z nazvu clanku ("Kdo byl Ludvík XIV.?", "Co je astronomie?"),
odpoved jsou prvni vety uvodu clanku. Nic se nestahuje, zadny cizi model.
Cast otazek je malymi pismeny a bez diakritiky, protoze tak lide casto pisou.

    python make_qa.py
"""
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
ZDROJ = ROOT / "data" / "wikipedia_cs.txt"
OUT = ROOT / "out"
KONEC = "<|endoftext|>"
KONTROLNICH = 2000

# (* 12. 3. 1950 ...) = zijici clovek, (1638 ... – 1715 ...) = zemrely clovek
ZIJICI = re.compile(r"\(\*\s")
ZEMRELY = re.compile(r"\([^()]*\b\d{3,4}\b[^()]*\s[–-]\s[^()]*\b\d{3,4}\b[^()]*\)")
MINULOST = re.compile(r"\s(byl|byla|bylo|byli|byly)\s")
PRITOMNOST = re.compile(r"\s(je|jsou)\s")
# konec vety: male pismeno nebo zavorka, tecka, mezera, velke pismeno
# (neroztrhne "Ludvík XIV." ani "angl. hobby"); deli se jen v mezere, takze
# tecka zustane u vety - jinak by se pri spojovani vet ztratila
KONEC_VETY = re.compile(r"(?<=[a-zěščřžýáíéůúťďňó\)]\.)\s+(?=[A-ZĚŠČŘŽÝÁÍÉŮÚŤĎŇÓ])")
PRESKOCIT = re.compile(r"^(Seznam|Rejstřík|Přehled)\b|^\d{1,4}$|^\d{1,2}\. \w+$")

OTAZKY_OSOBA = ["Kdo {byl} {x}?", "Kdo {byl} {x}?", "Kdo to {byl} {x}?", "{x} – kdo to {byl}?", "{x}?"]
OTAZKY_VEC = ["Co {je} {x}?", "Co {je} {x}?", "Co to {je} {x}?", "{x} – co to {je}?", "Vysvětli, co {je} {x}.", "{x}?"]


def clanky(cesta):
    buf = []
    with open(cesta, encoding="utf-8") as f:
        for radek in f:
            if radek.strip() == KONEC:
                yield "".join(buf).strip()
                buf = []
            else:
                buf.append(radek)


def bez_diakritiky(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def odpoved_z(uvod):
    # zbytky po odstranenych wiki znackach: "( a  Неветленфолу,  …)", "()"
    uvod = re.sub(r"\(\s*[,;]?\s*", "(", uvod)
    uvod = re.sub(r"\(\s*\)", "", uvod)
    uvod = re.sub(r"[ \t]{2,}", " ", uvod)
    vety = KONEC_VETY.split(uvod)
    text = ""
    for v in vety:
        text = f"{text} {v}".strip() if text else v.strip()
        if len(text) >= 100:
            break
    if not text.endswith((".", "!", "?")):
        text += "."
    if len(text) > 450:  # predlouha prvni veta - utnout na hranici slova
        text = text[:450].rsplit(" ", 1)[0].rstrip(",;") + "…"
    return text


def dvojice(clanek, rnd):
    radky = clanek.split("\n")
    nazev = radky[0].strip()
    uvod = "\n".join(radky[1:]).strip().split("\n\n")[0].strip()
    zaklad = re.sub(r"\s*\([^)]*\)\s*$", "", nazev).strip()  # "Merkur (planeta)" -> "Merkur"
    if (PRESKOCIT.search(nazev) or len(uvod) < 60 or not zaklad
            or not uvod.lower().startswith(zaklad[:4].lower())):
        return None

    zacatek = uvod[:300]
    osoba = bool(ZIJICI.search(zacatek) or ZEMRELY.search(zacatek))
    prvni_veta = KONEC_VETY.split(uvod)[0]
    minuly = bool(MINULOST.search(prvni_veta)) and not (
        ZIJICI.search(zacatek) or PRITOMNOST.search(prvni_veta))
    zena = " byla " in prvni_veta

    if osoba:
        byl = ("byla" if zena else "byl") if minuly or ZEMRELY.search(zacatek) else "je"
        x = zaklad
        if " " in zaklad and rnd.random() < 0.15:
            x = zaklad.split()[-1]  # jen prijmeni: "Kdo byl Hitler?"
        otazka = rnd.choice(OTAZKY_OSOBA).format(byl=byl, x=x)
    else:
        je = ("byla" if zena else "byl") if minuly else "je"
        otazka = rnd.choice(OTAZKY_VEC).format(je=je, x=zaklad)

    los = rnd.random()
    if los < 0.15:
        otazka = bez_diakritiky(otazka).lower()
    elif los < 0.35:
        otazka = otazka.lower()
    return {"otazka": otazka, "odpoved": odpoved_z(uvod), "osoba": osoba}


def main():
    rnd = random.Random(1234)
    vsechny, preskoceno = [], 0
    for i, c in enumerate(clanky(ZDROJ)):
        d = dvojice(c, rnd)
        if d:
            vsechny.append(d)
        else:
            preskoceno += 1
        if i % 50000 == 0:
            print(f"  {i:,} clanku", end="\r")
    rnd.shuffle(vsechny)

    OUT.mkdir(exist_ok=True)
    for nazev, cast in (("qa_val.jsonl", vsechny[:KONTROLNICH]),
                        ("qa_train.jsonl", vsechny[KONTROLNICH:])):
        with open(OUT / nazev, "w", encoding="utf-8") as f:
            for d in cast:
                f.write(json.dumps({"otazka": d["otazka"], "odpoved": d["odpoved"]},
                                   ensure_ascii=False) + "\n")

    osob = sum(d["osoba"] for d in vsechny)
    print(f"\n{len(vsechny):,} dvojic ({osob:,} o lidech), {preskoceno:,} clanku preskoceno")
    print(f"out/qa_train.jsonl: {len(vsechny) - KONTROLNICH:,}  |  out/qa_val.jsonl: {KONTROLNICH:,}")
    print("\nukazky:")
    for d in vsechny[:8]:
        print(f"  {d['otazka']}\n    -> {d['odpoved'][:140]}")


if __name__ == "__main__":
    main()

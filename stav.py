"""
Kolik procent treninku je hotovo a kolik zbyva.

    python stav.py                 # z Windows i z Linuxu
    python stav.py cesta/k/trenink.log
"""
import re
import sys
import time
from pathlib import Path

KROK = re.compile(r"^krok\s+(\d+)/(\d+).*?([\d.]+)s/krok")
VAL = re.compile(r">> val loss ([\d.]+)\s+\(perplexita ([\d.]+)\)")


def main():
    log = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "trenink.log")
    if not log.exists():
        raise SystemExit(f"log neexistuje: {log}")

    # PowerShell drive zapisoval log v UTF-16, novy zapis je UTF-8 - zvladni obe
    raw = log.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    lines = text.replace("\r", "").splitlines()

    kroky = [(int(m[1]), int(m[2]), float(m[3])) for m in map(KROK.match, lines) if m]
    if not kroky:
        raise SystemExit("v logu zatim neni zadny krok")
    krok, celkem, _ = kroky[-1]

    # Tempo jen z aktualniho behu - starsi behy mohly jet jinak rychle
    # (treba s hrou na pozadi). Pokud je jich od navazani moc malo, vezme
    # se i predchozi historie, at je z ceho pocitat.
    start = max((i for i, l in enumerate(lines) if "navazuji od kroku" in l), default=0)
    aktualni = [(int(m[1]), int(m[2]), float(m[3]))
                for m in map(KROK.match, lines[start:]) if m]
    vzorek = aktualni if len(aktualni) >= 3 else kroky
    # Median: prvni radek po navazani zapocitava i start a muze ukazat
    # treba 460 s/krok - jeden takovy ulet by odhad nafoukl na tisice hodin.
    posledni = sorted(s for _, _, s in vzorek[-20:])
    s_krok = posledni[len(posledni) // 2]
    val = [m for m in map(VAL.search, lines) if m]

    stari = (time.time() - log.stat().st_mtime) / 60
    zbyva_h = (celkem - krok) * s_krok / 3600

    sirka = 30
    plne = round(sirka * krok / celkem)
    print(f"[{'#' * plne}{'.' * (sirka - plne)}] {100 * krok / celkem:.1f} %")
    print(f"krok {krok:,} z {celkem:,}   tempo {s_krok:.1f} s/krok")
    print(f"zbyva priblizne {zbyva_h:.0f} h "
          f"(hotovo kolem {time.strftime('%d.%m. %H:%M', time.localtime(time.time() + zbyva_h * 3600))})")
    if val:
        print(f"posledni kontrola: perplexita {val[-1][2]} (mensi = lepsi)")
    if stari > 15:
        print(f"\n! log se nezmenil {stari:.0f} min - trenink nejspis nebezi")


if __name__ == "__main__":
    main()

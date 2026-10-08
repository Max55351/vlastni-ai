"""
Vlastni AI - okno pro psani s vlastnim modelem.

Spousti se z ikony na plose. Pripona .pyw = bez cerneho okna konzole.
Dva rezimy:
  Ptat se             - out/model_qa.pt: tvuj model doucenty odpovidat na otazky
  Pokracovat v textu  - out/model.pt: tvuj model od nuly, pokracuje v zadanem textu
"""
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Bez konzole (pythonw) by se chyby nikam nevypsaly a program by jen potichu
# zmizel. Vsechno proto jde do program.log - vcetne tvrdych padu (faulthandler).
if sys.stderr is None or sys.stdout is None:
    _log = open(os.path.join(ROOT, "program.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log
    import datetime
    import faulthandler
    import threading as _th
    import traceback as _tb
    print(f"\n=== start {datetime.datetime.now():%d.%m.%Y %H:%M:%S} ===", file=_log)
    faulthandler.enable(file=_log, all_threads=True)
    sys.excepthook = lambda t, v, tb: _tb.print_exception(t, v, tb, file=_log)
    _th.excepthook = lambda a: _tb.print_exception(a.exc_type, a.exc_value, a.exc_traceback, file=_log)

KONEC_CLANKU = "<|endoftext|>"
QA_MODEL = os.path.join(ROOT, "out", "model_qa.pt")

REZIMY = {
    "otazky": dict(popis="Otázka:", napoveda="Kdo byl Napoleon Bonaparte?", tlacitko="Zeptat se  ▶",
                   pozn="Odpovídá tvůj model. Fakta si může plést.  ·  Ctrl+Enter = Zeptat se"),
    "text": dict(popis="Začátek textu (model v něm bude pokračovat):", napoveda="Napoleon Bonaparte byl",
                 tlacitko="Napiš  ▶",
                 pozn="Pokračuje v textu, na otázky neodpovídá. Fakta si může vymýšlet.  ·  Ctrl+Enter = Napiš"),
}


class Aplikace:
    def __init__(self, okno):
        self.okno = okno
        self.fronta = queue.Queue()
        self.stop = threading.Event()
        self.modely = {}          # "text" / "otazky" -> nacteny model
        self.tok = self.device = None
        self.sablona = None       # sablona otazky - musi sedet s tou z doladeni
        self.generuje = False

        okno.title("Vlastní AI")
        okno.geometry("900x680")
        okno.minsize(600, 460)
        self._postav_okno()

        self._stav("Načítám model…")
        threading.Thread(target=self._nacti_model, daemon=True).start()
        okno.after(40, self._zpracuj_frontu)

    # ------------------------------------------------------------ vzhled

    def _postav_okno(self):
        o = self.okno
        o.columnconfigure(0, weight=1)
        o.rowconfigure(2, weight=1)
        pismo = ("Segoe UI", 10)

        # --- zadani ---
        horni = ttk.Frame(o, padding=(12, 12, 12, 4))
        horni.grid(row=0, column=0, sticky="nsew")
        horni.columnconfigure(1, weight=1)

        self.rezim = tk.StringVar(value="otazky" if os.path.exists(QA_MODEL) else "text")
        prepinac = ttk.Frame(horni)
        prepinac.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(prepinac, text="Režim:", font=pismo).pack(side="left", padx=(0, 8))
        self.r_otazky = ttk.Radiobutton(prepinac, text="Ptát se", value="otazky",
                                        variable=self.rezim, command=self._zmena_rezimu)
        self.r_otazky.pack(side="left", padx=(0, 12))
        ttk.Radiobutton(prepinac, text="Pokračovat v textu", value="text",
                        variable=self.rezim, command=self._zmena_rezimu).pack(side="left")
        self.l_qa = ttk.Label(prepinac, text="", font=("Segoe UI", 9), foreground="#888888")
        self.l_qa.pack(side="left", padx=(12, 0))

        self.l_zadani = ttk.Label(horni, text="", font=pismo)
        self.l_zadani.grid(row=1, column=0, columnspan=2, sticky="w")
        self.zadani = tk.Text(horni, height=3, wrap="word", font=("Segoe UI", 11),
                              relief="solid", borderwidth=1, padx=6, pady=4)
        self.zadani.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.zadani.bind("<Control-Return>", lambda e: (self.napis(), "break")[1])
        self.zadani.focus_set()

        # --- ovladani ---
        ovl = ttk.Frame(o, padding=(12, 4, 12, 4))
        ovl.grid(row=1, column=0, sticky="ew")

        self.delka = tk.IntVar(value=160)
        self.kreativita = tk.DoubleVar(value=0.7)
        self.zastavit = tk.BooleanVar(value=True)
        self.bez_opakovani = tk.BooleanVar(value=True)

        self._posuvnik(ovl, 0, "Délka", self.delka, 40, 400, lambda v: f"{int(float(v))} tokenů")
        self._posuvnik(ovl, 1, "Kreativita", self.kreativita, 0.3, 1.2, lambda v: f"{float(v):.2f}")

        volby = ttk.Frame(ovl)
        volby.grid(row=0, column=3, rowspan=2, sticky="w", padx=(18, 0))
        self.c_zastavit = ttk.Checkbutton(volby, text="Zastavit na konci článku",
                                          variable=self.zastavit)
        self.c_zastavit.pack(anchor="w")
        ttk.Checkbutton(volby, text="Méně opakování",
                        variable=self.bez_opakovani).pack(anchor="w")

        tlac = ttk.Frame(ovl)
        tlac.grid(row=0, column=4, rowspan=2, sticky="e", padx=(18, 0))
        ovl.columnconfigure(4, weight=1)
        self.b_napis = ttk.Button(tlac, text="Napiš  ▶", command=self.napis, state="disabled")
        self.b_dal = ttk.Button(tlac, text="Pokračovat dál", command=self.pokracuj, state="disabled")
        self.b_stop = ttk.Button(tlac, text="Stop", command=self.stop.set, state="disabled")
        self.b_smaz = ttk.Button(tlac, text="Vymazat", command=self.vymaz)
        for i, b in enumerate((self.b_napis, self.b_dal, self.b_stop, self.b_smaz)):
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)

        # --- vystup ---
        self.vystup = ScrolledText(o, wrap="word", font=("Segoe UI", 11),
                                   relief="solid", borderwidth=1, padx=10, pady=8)
        self.vystup.grid(row=2, column=0, sticky="nsew", padx=12, pady=(4, 4))
        self.vystup.tag_configure("zadani", font=("Segoe UI", 11, "bold"), foreground="#1f4e79")
        self.vystup.tag_configure("pozn", foreground="#888888", font=("Segoe UI", 9, "italic"))

        # --- stavovy radek ---
        dolni = ttk.Frame(o, padding=(12, 0, 12, 10))
        dolni.grid(row=3, column=0, sticky="ew")
        dolni.columnconfigure(0, weight=1)
        self.l_stav = ttk.Label(dolni, text="", font=("Segoe UI", 9))
        self.l_stav.grid(row=0, column=0, sticky="w")
        self.l_pozn = ttk.Label(dolni, font=("Segoe UI", 9), foreground="#888888")
        self.l_pozn.grid(row=0, column=1, sticky="e")

        self._zmena_rezimu(prvni=True)

    def _posuvnik(self, rodic, radek, nazev, promenna, od, do, format_):
        ttk.Label(rodic, text=nazev, width=10).grid(row=radek, column=0, sticky="w")
        hodnota = ttk.Label(rodic, width=11)
        s = ttk.Scale(rodic, from_=od, to=do, variable=promenna, length=200,
                      command=lambda v: hodnota.configure(text=format_(v)))
        s.grid(row=radek, column=1, sticky="w", pady=2)
        hodnota.grid(row=radek, column=2, sticky="w", padx=(6, 0))
        hodnota.configure(text=format_(promenna.get()))

    def _zmena_rezimu(self, prvni=False):
        r = REZIMY[self.rezim.get()]
        stary = self.zadani.get("1.0", "end-1c").strip()
        self.l_zadani.configure(text=r["popis"])
        self.l_pozn.configure(text=r["pozn"])
        self.b_napis.configure(text=r["tlacitko"])
        # prazdne pole nebo nezmenena napoveda druheho rezimu -> nova napoveda
        if prvni or not stary or stary in (x["napoveda"] for x in REZIMY.values()):
            self.zadani.delete("1.0", "end")
            self.zadani.insert("1.0", r["napoveda"])
            self.zadani.tag_add("sel", "1.0", "end-1c")
        # odpoved konci vzdy sama - zaskrtavatko ma smysl jen u textu
        self.c_zastavit.configure(state="normal" if self.rezim.get() == "text" else "disabled")
        self._tlacitka()

    def _stav(self, text):
        self.l_stav.configure(text=text)

    def _tlacitka(self):
        pripraveno = self.rezim.get() in self.modely
        self.b_napis.configure(state="normal" if pripraveno and not self.generuje else "disabled")
        dal = pripraveno and not self.generuje and self.rezim.get() == "text"
        self.b_dal.configure(state="normal" if dal else "disabled")
        self.b_stop.configure(state="normal" if self.generuje else "disabled")
        self.b_smaz.configure(state="disabled" if self.generuje else "normal")

    # ------------------------------------------------------------ akce

    def napis(self):
        zadano = self.zadani.get("1.0", "end-1c").strip()
        rezim = self.rezim.get()
        if not zadano:
            messagebox.showinfo("Vlastní AI", f"Napiš nejdřív něco, třeba „{REZIMY[rezim]['napoveda']}“.")
            return
        if rezim not in self.modely or self.generuje:
            return
        if rezim == "otazky":
            # otazky se skladaji pod sebe jako rozhovor; model si predchozi
            # otazky nepamatuje, kazda se posila samostatne
            if self.vystup.get("1.0", "end-1c").strip():
                self.vystup.insert("end", "\n\n")
            self.vystup.insert("end", zadano + "\n", "zadani")
            self.vystup.see("end")
            self._spust(self.sablona.format(q=zadano), min_tokenu=3, zastavit=True)
        else:
            self.vystup.delete("1.0", "end")
            self.vystup.insert("end", zadano, "zadani")
            self._spust(zadano, min_tokenu=20, zastavit=self.zastavit.get())

    def pokracuj(self):
        text = self.vystup.get("1.0", "end-1c")
        if not text.strip():
            self.napis()
            return
        if "text" not in self.modely or self.generuje:
            return
        # text v okne casto konci koncem clanku - bez minimalni delky by
        # model hned clanek ukoncil a nic nepridal
        self._spust(text, min_tokenu=40, zastavit=self.zastavit.get())

    def vymaz(self):
        self.vystup.delete("1.0", "end")

    def _spust(self, text, min_tokenu, zastavit):
        self.stop.clear()
        self.generuje = True
        self._tlacitka()
        self._stav("Píšu…")
        threading.Thread(
            target=self._generuj,
            args=(self.modely[self.rezim.get()], text, int(self.delka.get()),
                  float(self.kreativita.get()), zastavit,
                  1.15 if self.bez_opakovani.get() else 1.0, min_tokenu),
            daemon=True,
        ).start()

    # ------------------------------------------------------------ prace na pozadi

    def _nacti_model(self):
        try:
            import torch
            from chat import load
            model, tok, device = load()
            modely = {"text": model}
            sablona = None
            if os.path.exists(QA_MODEL):
                from finetune_vlastni import SABLONA
                modely["otazky"], _, _ = load("model_qa.pt")
                sablona = SABLONA
            gpu = (torch.cuda.get_device_name(0) if device == "cuda"
                   else "procesor (grafika nenalezena, bude to pomalé)")
            info = f"Model připraven  ·  {model.num_params() / 1e6:.1f} M parametrů  ·  {gpu}"
            self.fronta.put(("pripraven", (modely, tok, device, sablona, info)))
        except OSError as e:
            if "4551" in str(e):
                self.fronta.put(("chyba",
                    "Windows (Smart App Control) zablokoval knihovnu PyTorch, "
                    "takže model nejde načíst.\n\nNěkdy to Windows po čase zase pustí. "
                    "Spolehlivé řešení je spouštět model přes WSL - postup je v README."))
            else:
                self.fronta.put(("chyba", f"Model se nepodařilo načíst:\n{e}"))
        except BaseException as e:  # vcetne SystemExit z load(), kdyz chybi checkpoint
            self.fronta.put(("chyba", f"Model se nepodařilo načíst:\n{e}"))

    def _generuj(self, model, text, delka, teplota, zastavit, penalizace, min_tokenu):
        import torch
        try:
            ids = self.tok.encode(text) or [0]
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            konec_id = self.tok.special.get(KONEC_CLANKU)
            stop_ids = (konec_id,) if zastavit and konec_id is not None else ()

            nove, vypsano = [], ""

            def na_token(t):
                nonlocal vypsano
                if t == konec_id:
                    return
                nove.append(t)
                hotovy = self.tok.decode(nove)
                # pismena s diakritikou maji 2 bajty a muzou byt rozpulena
                # mezi dva tokeny - s vypsanim se pocka na druhou pulku
                if hotovy.endswith("�"):
                    return
                self.fronta.put(("text", hotovy[len(vypsano):]))
                vypsano = hotovy

            model.generate(
                x, delka, temperature=teplota, top_k=50, top_p=0.95,
                repetition_penalty=penalizace, stop_ids=stop_ids,
                min_new_tokens=min(min_tokenu, delka),
                on_token=na_token, should_stop=self.stop.is_set,
            )
            zbytek = self.tok.decode(nove)[len(vypsano):]
            if zbytek:
                self.fronta.put(("text", zbytek))
            self.fronta.put(("hotovo", "zastaveno" if self.stop.is_set() else "hotovo"))
        except Exception as e:
            self.fronta.put(("chyba", f"Psaní selhalo:\n{e}"))
            self.fronta.put(("hotovo", "chyba"))

    def _zpracuj_frontu(self):
        try:
            while True:
                druh, data = self.fronta.get_nowait()
                if druh == "text":
                    self.vystup.insert("end", data)
                    self.vystup.see("end")
                elif druh == "pripraven":
                    self.modely, self.tok, self.device, self.sablona, info = data
                    if "otazky" not in self.modely:
                        self.r_otazky.configure(state="disabled")
                        self.l_qa.configure(text="(ptaní ještě není naučené)")
                        if self.rezim.get() == "otazky":
                            self.rezim.set("text")
                            self._zmena_rezimu()
                    self._stav(info)
                    self._tlacitka()
                elif druh == "hotovo":
                    self.generuje = False
                    self._tlacitka()
                    if data == "zastaveno":
                        self.vystup.insert("end", "  [zastaveno]", "pozn")
                    self._stav("Hotovo." if data == "hotovo" else
                               "Zastaveno." if data == "zastaveno" else "Něco se pokazilo.")
                elif druh == "chyba":
                    self._stav("Chyba - viz zpráva.")
                    messagebox.showerror("Vlastní AI", data)
        except queue.Empty:
            pass
        self.okno.after(40, self._zpracuj_frontu)


def main():
    try:  # ostre pismo na monitorech se zvetsenim
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:  # vlastni identita na liste - jinak se okno schova pod ikonu Pythonu
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VlastniAI.Program")
    except Exception:
        pass
    okno = tk.Tk()
    try:
        okno.iconbitmap(default=os.path.join(ROOT, "launcher", "ikona.ico"))
    except tk.TclError:
        pass  # bez ikony to pobezi taky
    Aplikace(okno)
    okno.mainloop()


if __name__ == "__main__":
    main()

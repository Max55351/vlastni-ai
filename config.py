"""
Presety velikosti modelu.

Cisla VRAM nize jsou zmerena na teto masine (RTX 3050, bf16, plny slovnik),
ne odhadnuta. Autodetekce vybere nejvetsi preset, ktery se do karty realne vejde
- vcetne rezervy, aby do ni slo zaroven neco delat.
"""
import torch

from model import Config

PRESETS = {
    # ~11M - overeni, ze pipeline funguje. Minuty treninku.
    "tiny": dict(
        model=Config(vocab_size=4096, n_layer=6, n_head=6, n_embd=384, block_size=256),
        batch_size=16, grad_accum=2, lr=1e-3, max_steps=3000,
        vram_gb=0.8,
    ),
    # ~30M - bezpecna volba pro 4-6 GB karty.
    "small": dict(
        model=Config(vocab_size=8192, n_layer=8, n_head=8, n_embd=512, block_size=512),
        batch_size=8, grad_accum=8, lr=6e-4, max_steps=20000,
        vram_gb=1.75,
    ),
    # ~98M (velikost GPT-2 small) - na 6 GB karte jede v pohode.
    "medium": dict(
        model=Config(vocab_size=16384, n_layer=12, n_head=12, n_embd=768, block_size=512),
        batch_size=6, grad_accum=12, lr=5e-4, max_steps=30000,
        vram_gb=3.41,
    ),
    # ~320M - potrebuje 12 GB+. Na mensi karte to Windows preleje do systemove
    # RAM: nespadne to, ale trenink bude nekolikanasobne pomalejsi.
    "large": dict(
        model=Config(vocab_size=16384, n_layer=24, n_head=16, n_embd=1024, block_size=1024),
        batch_size=4, grad_accum=24, lr=3e-4, max_steps=40000,
        vram_gb=10.45,
    ),
}

HEADROOM_GB = 1.5  # rezerva na displej, prohlizec a fragmentaci alokatoru


def detect_preset():
    if not torch.cuda.is_available():
        return "tiny"
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    for name in ("large", "medium", "small", "tiny"):
        if PRESETS[name]["vram_gb"] + HEADROOM_GB <= gb:
            return name
    return "tiny"


def get(name=None):
    name = name or detect_preset()
    if name not in PRESETS:
        raise SystemExit(f"neznamy preset {name!r}, dostupne: {list(PRESETS)}")
    return name, PRESETS[name]

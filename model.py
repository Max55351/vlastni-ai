"""
Vlastni jazykovy model - transformer (decoder-only, GPT-style) od nuly.
Zadne predtrenovane vahy, zadna cizi knihovna krome PyTorche.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256      # delka kontextu v tokenech
    dropout: float = 0.1
    bias: bool = False         # bias v Linear/LayerNorm; bez nej to trenuje o kus lip


class RMSNorm(nn.Module):
    """Normalizace bez odecitani prumeru - levnejsi a stabilnejsi nez LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_cache(head_dim: int, max_len: int, device, base: float = 10000.0):
    """Rotacni pozicni embeddingy - pozice se koduje rotaci vektoru, ne pricitanim."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim)
    T = x.size(2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SelfAttention(nn.Module):
    """Kauzalni multi-head self-attention. Kazdy token vidi jen sebe a to pred sebou."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # flash attention pres PyTorch - setri pamet, coz je na 6 GB zasadni
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class SwiGLU(nn.Module):
    """Feed-forward vrstva se SwiGLU aktivaci."""

    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(8 / 3 * cfg.n_embd)
        hidden = 64 * ((hidden + 63) // 64)  # zarovnani na nasobek 64 kvuli rychlosti
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.n2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # sdilene vahy vstup/vystup

        self.apply(self._init_weights)
        # skalovani vah v rezidualnich cestach podle hloubky site
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self._rope_device = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, device):
        if self._rope_device != device:
            cos, sin = build_rope_cache(
                self.cfg.n_embd // self.cfg.n_head, self.cfg.block_size, device
            )
            self.register_buffer("_cos", cos, persistent=False)
            self.register_buffer("_sin", sin, persistent=False)
            self._rope_device = device
        return self._cos, self._sin

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"kontext {T} > block_size {self.cfg.block_size}"
        cos, sin = self._rope(idx.device)

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)

        if targets is None:
            # pri generovani staci spocitat logity jen pro posledni pozici
            return self.head(x[:, -1:, :]), None

        logits = self.head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1.0, penalty_window=128, stop_ids=(),
                 min_new_tokens=0, on_token=None, should_stop=None):
        """
        repetition_penalty > 1 snizi sanci tokenu, ktere se objevily v poslednich
        penalty_window tokenech - maly model se jinak rad zacykli.
        stop_ids: generovani skonci po vygenerovani nektereho z nich (konec clanku).
        min_new_tokens: tolik tokenu se vygeneruje vzdycky - do te doby se stop_ids
            nesmi objevit. Jinak by pokracovani textu, ktery konci koncem clanku,
            skoncilo hned prvnim tokenem a nic nepridalo.
        on_token(id): zavola se po kazdem novem tokenu (prubezne vypisovani).
        should_stop(): kdyz vrati True, generovani skonci (tlacitko Stop).
        """
        self.eval()
        for step in range(max_new_tokens):
            if should_stop is not None and should_stop():
                break
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()

            if stop_ids and step < min_new_tokens:
                logits[:, list(stop_ids)] = float("-inf")

            if repetition_penalty != 1.0:
                prev = idx[:, -penalty_window:]
                score = torch.gather(logits, 1, prev)
                score = torch.where(score > 0, score / repetition_penalty,
                                    score * repetition_penalty)
                logits = logits.scatter(1, prev, score)

            logits = logits / max(temperature, 1e-5)

            if top_k:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k).values[..., -1, None]
                logits = logits.masked_fill(logits < thresh, float("-inf"))

            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            t = int(nxt[0, 0])
            if on_token is not None:
                on_token(t)
            if t in stop_ids:
                break
        return idx

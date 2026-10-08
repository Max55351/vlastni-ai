"""
Vlastni byte-level BPE tokenizer - natrenovany na tvych datech, od nuly.
Zadny cizi slovnik. Diky byte-level zakladu zvladne cokoliv vcetne cestiny a emoji.
"""
import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self):
        self.merges = {}       # (a, b) -> novy_token_id
        self.vocab = {}        # id -> bytes
        self.special = {}      # "<|endoftext|>" -> id
        self._by_id = {}       # novy_token_id -> (a, b), obracena mapa pro encode
        self._cache = {}       # slovo -> tokeny, stejna slova se opakuji porad dokola

    # ---------- trenovani ----------

    def train(self, text: str, vocab_size: int, specials=("<|endoftext|>",), verbose=True):
        assert vocab_size > 256 + len(specials)
        target = vocab_size - len(specials)

        # slova drzime oddelene, aby merge nikdy neprekrocil hranici mezery
        words = Counter()
        for chunk in _split_words(text):
            words[tuple(chunk.encode("utf-8"))] += 1
        seqs = [[list(w), c] for w, c in words.items()]
        if verbose:
            print(f"  {len(seqs):,} unikatnich slov")

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        next_id = 256

        # Statistiky se nepocitaji znovu po kazdem sloucení - to by na velkem
        # korpusu trvalo dny. Drzime prubezny pocet vyskytu kazde dvojice plus
        # index, ve kterych slovech se dvojice vyskytuje, a po slouceni
        # prepocitame jen ta dotcena slova.
        pairs = Counter()
        where = {}  # (a, b) -> mnozina indexu slov
        for i, (seq, count) in enumerate(seqs):
            for pair in zip(seq, seq[1:]):
                pairs[pair] += count
                where.setdefault(pair, set()).add(i)

        while next_id < target:
            if not pairs:
                break
            best, freq = max(pairs.items(), key=lambda kv: kv[1])
            if freq < 2:
                break

            self.merges[best] = next_id
            self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]

            for i in list(where.get(best, ())):
                seq, count = seqs[i]
                if best not in set(zip(seq, seq[1:])):
                    continue
                # odecti stare dvojice tohoto slova
                for pair in zip(seq, seq[1:]):
                    pairs[pair] -= count
                    if pairs[pair] <= 0:
                        del pairs[pair]
                        where.pop(pair, None)
                    else:
                        where[pair].discard(i)
                new = _merge(seq, best, next_id)
                seqs[i][0] = new
                # a pricti nove
                for pair in zip(new, new[1:]):
                    pairs[pair] += count
                    where.setdefault(pair, set()).add(i)

            pairs.pop(best, None)
            where.pop(best, None)

            if verbose and (next_id - 256) % 500 == 0:
                shown = self.vocab[next_id].decode("utf-8", errors="replace")
                print(f"  merge {next_id - 256:5d}/{target - 256}  "
                      f"{freq:7d}x  {shown!r}")
            next_id += 1

        for i, tok in enumerate(specials):
            self.special[tok] = next_id + i
            self.vocab[next_id + i] = tok.encode("utf-8")

        if verbose:
            print(f"  hotovo: slovnik ma {len(self.vocab)} tokenu")
        return self

    # ---------- pouziti ----------

    def encode(self, text: str, allow_special=True):
        if allow_special and self.special:
            out = []
            parts = [text]
            for tok, tid in self.special.items():
                new = []
                for p in parts:
                    if isinstance(p, int):
                        new.append(p)
                        continue
                    pieces = p.split(tok)
                    for i, piece in enumerate(pieces):
                        if i:
                            new.append(tid)
                        if piece:
                            new.append(piece)
                parts = new
            for p in parts:
                out.extend([p] if isinstance(p, int) else self._encode_plain(p))
            return out
        return self._encode_plain(text)

    def _encode_plain(self, text: str):
        ids = []
        for chunk in _split_words(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_word(chunk)
                self._cache[chunk] = cached
            ids.extend(cached)
        return ids

    def _encode_word(self, chunk: str):
        if not self._by_id:
            self._by_id = {i: p for p, i in self.merges.items()}
        seq = list(chunk.encode("utf-8"))
        while len(seq) >= 2:
            # vzdy aplikuj merge, ktery vznikl nejdriv (nejnizsi id)
            cand = min(
                (self.merges[p] for p in zip(seq, seq[1:]) if p in self.merges),
                default=None,
            )
            if cand is None:
                break
            seq = _merge(seq, self._by_id[cand], cand)
        return seq

    def decode(self, ids):
        data = b"".join(self.vocab[int(i)] for i in ids if int(i) in self.vocab)
        return data.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ---------- ulozeni ----------

    def save(self, path):
        Path(path).write_text(json.dumps({
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
            "special": self.special,
            "vocab_size": len(self.vocab),
        }), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        t = cls()
        t.merges = {(a, b): i for a, b, i in data["merges"]}
        t._by_id = {i: p for p, i in t.merges.items()}
        t.special = data["special"]
        t.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in sorted(t.merges.items(), key=lambda kv: kv[1]):
            t.vocab[i] = t.vocab[a] + t.vocab[b]
        for tok, i in t.special.items():
            t.vocab[i] = tok.encode("utf-8")
        return t


def _merge(seq, pair, new_id):
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _split_words(text: str):
    """Rozdeli text na kousky tak, aby mezera patrila k nasledujicimu slovu."""
    buf = ""
    for ch in text:
        if ch.isspace():
            if buf and not buf[-1].isspace():
                yield buf
                buf = ch
            else:
                buf += ch
        else:
            if buf and buf[-1].isspace() and len(buf) > 1:
                yield buf[:-1]
                buf = buf[-1] + ch
            else:
                buf += ch
    if buf:
        yield buf

import json
import re

from lm.bpe import _PAT


def save_tokenizer(vocab, merges, vocab_path, merges_path):
    with open(vocab_path, "w") as f:
        json.dump({str(i): b.hex() for i, b in vocab.items()}, f)
    with open(merges_path, "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.special_tokens = list(special_tokens or [])
        self.inv = {b: i for i, b in self.vocab.items()}
        for s in self.special_tokens:
            sb = s.encode("utf-8")
            if sb not in self.inv:
                nid = len(self.vocab)
                self.vocab[nid] = sb
                self.inv[sb] = nid
        self.ranks = {pair: r for r, pair in enumerate(merges)}
        self.cache = {}
        self._specials = set(self.special_tokens)
        if self.special_tokens:
            specs = sorted(self.special_tokens, key=len, reverse=True)
            self._split = re.compile("(" + "|".join(map(re.escape, specs)) + ")")
        else:
            self._split = None

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path) as f:
            vocab = {int(i): bytes.fromhex(h) for i, h in json.load(f).items()}
        merges = []
        with open(merges_path) as f:
            for line in f:
                if line.strip():
                    a, b = line.split()
                    merges.append((bytes.fromhex(a), bytes.fromhex(b)))
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, tok: bytes):
        ids = self.cache.get(tok)
        if ids is not None:
            return ids
        parts = [bytes([x]) for x in tok]
        while len(parts) > 1:
            best, best_rank = None, None
            for p in zip(parts, parts[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            a, b = best
            new, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(parts[i])
                    i += 1
            parts = new
        ids = [self.inv[p] for p in parts]
        if len(self.cache) > 200_000:
            self.cache.clear()
        self.cache[tok] = ids
        return ids

    def _encode_gen(self, text: str):
        pieces = self._split.split(text) if self._split else [text]
        for piece in pieces:
            if not piece:
                continue
            if piece in self._specials:
                yield self.inv[piece.encode("utf-8")]
            else:
                for m in _PAT.finditer(piece):
                    yield from self._encode_pretoken(m.group().encode("utf-8"))

    def encode(self, text: str):
        return list(self._encode_gen(text))

    def encode_iterable(self, iterable):
        for chunk in iterable:
            yield from self._encode_gen(chunk)

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
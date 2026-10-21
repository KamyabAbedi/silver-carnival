import re
from collections import Counter, defaultdict

import regex 

import os
import multiprocessing as mp


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = regex.compile(PAT)


def count_pretokens(text: str, special_tokens: list[str]) -> Counter:
    counts: Counter = Counter()

    if special_tokens:
        split_pat = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))
        chunks = re.split(split_pat, text)
    else:
        chunks = [text]  # an empty pattern would split between every character

    for chunk in chunks:
        for m in _PAT.finditer(chunk):
            counts[m.group().encode("utf-8")] += 1
    return counts


def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode("utf-8")

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        text = f.read()

    counts = count_pretokens_file(input_path, special_tokens)

    words = [[bytes([b]) for b in w] for w in counts]
    freqs = list(counts.values())
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            pair_to_words[p].add(wi)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        a, b = best
        new = a + b
        merges.append(best)
        vocab[len(vocab)] = new

        touched = {best}
        for wi in list(pair_to_words[best]):
            w, f = words[wi], freqs[wi]

            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                touched.add(p)

            new_w, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    new_w.append(new)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            words[wi] = new_w

            for p in zip(new_w, new_w[1:]):
                pair_counts[p] += f
                pair_to_words[p].add(wi)
                touched.add(p)

        for p in touched:
            if pair_counts.get(p, 0) <= 0:
                pair_counts.pop(p, None)
        del pair_to_words[best]

    return vocab, merges




def _find_boundaries(path, num_chunks, split_token=b"<|endoftext|>"):
    size = os.path.getsize(path)
    approx = max(size // num_chunks, 1)
    bounds = [i * approx for i in range(num_chunks)] + [size]
    with open(path, "rb") as f:
        for i in range(1, len(bounds) - 1):
            pos = bounds[i]
            while True:
                f.seek(pos)
                buf = f.read(4096)
                if not buf:
                    pos = size
                    break
                j = buf.find(split_token)
                if j != -1:
                    pos += j
                    break
                pos += 4096 - (len(split_token) - 1)
            bounds[i] = pos
    return sorted(set(bounds))


def _count_chunk(args):
    path, start, end, specials = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return count_pretokens(text, specials)


def count_pretokens_file(path, special_tokens, num_workers=None):
    num_workers = num_workers or os.cpu_count() or 1
    bounds = _find_boundaries(path, num_workers * 4)
    jobs = [(path, s, e, special_tokens) for s, e in zip(bounds, bounds[1:]) if e > s]
    total = Counter()
    if len(jobs) <= 1:
        for j in jobs:
            total.update(_count_chunk(j))
        return total
    with mp.Pool(num_workers) as pool:
        for c in pool.imap_unordered(_count_chunk, jobs):
            total.update(c)
    return total
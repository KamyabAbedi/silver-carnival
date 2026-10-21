import time
from lm.tokenizer import Tokenizer

tok = Tokenizer.from_files("tok/vocab.json", "tok/merges.txt", ["<|endoftext|>"])

with open("tok/merges.txt") as f:
    for line in list(f)[:20]:
        a, b = line.split()
        print(bytes.fromhex(a), bytes.fromhex(b))

with open("data/tinystories_valid.txt", "rb") as f:
    raw = f.read(15_000_000)
text = raw.decode("utf-8", errors="ignore")
t0 = time.perf_counter()
ids = tok.encode(text)
dt = time.perf_counter() - t0
nbytes = len(text.encode("utf-8"))
print("bytes/token:", nbytes / len(ids))
print("encode speed:", nbytes / 1e6 / dt, "MB/s")
# scripts/cut_200mb.py
src, dst, limit = "data/tinystories_train.txt", "data/ts_200mb.txt", 200_000_000
tok = b"<|endoftext|>"
with open(src, "rb") as f:
    data = f.read(limit)
cut = data.rfind(tok) + len(tok)
with open(dst, "wb") as f:
    f.write(data[:cut])
print("wrote", cut, "bytes")
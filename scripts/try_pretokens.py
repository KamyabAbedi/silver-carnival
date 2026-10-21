from lm.bpe import count_pretokens

path = "data/tinystories_valid.txt"  # adjust to where your file lives
with open(path, "rb") as f:
    raw = f.read(1_000_000)
text = raw.decode("utf-8", errors="ignore")  # drops a character cut in half at the 1 MB edge

counts = count_pretokens(text, ["<|endoftext|>"])
print("distinct pre-tokens:", len(counts), " total:", sum(counts.values()))
for tok, c in counts.most_common(10):
    print(c, tok)
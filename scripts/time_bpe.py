import pickle, resource, sys, time
from lm.bpe import train_bpe, count_pretokens_file

SPECIALS = ["<|endoftext|>"]

if __name__ == "__main__":
    path = "data/ts_200mb.txt"

    t0 = time.perf_counter()
    counts = count_pretokens_file(path, SPECIALS)
    t_count = time.perf_counter() - t0
    print(f"counting: {t_count:.1f}s, distinct pre-tokens: {len(counts)}")

    t0 = time.perf_counter()
    vocab, merges = train_bpe(path, 10_000, SPECIALS)
    t_total = time.perf_counter() - t0
    print(f"train_bpe total: {t_total:.1f}s  (merge loop ≈ {t_total - t_count:.1f}s)")

    scale = 1e6 if sys.platform == "darwin" else 1e3
    main_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
    kids_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / scale
    print(f"peak RSS main: {main_mb:.0f} MB, largest worker: {kids_mb:.0f} MB")

    longest = max(vocab.values(), key=len)
    print("longest token:", longest, len(longest), "bytes")

    with open("data/bpe_10k.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)
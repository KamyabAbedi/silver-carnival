import os
import pickle
from lm.tokenizer import save_tokenizer

vocab, merges = pickle.load(open("data/bpe_10k.pkl", "rb"))
os.makedirs("tok", exist_ok=True)
save_tokenizer(vocab, merges, "tok/vocab.json", "tok/merges.txt")
print("saved")
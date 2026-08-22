import struct
from transformers import GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2-medium")

with open("model/input.txt", "r") as f:
    text = f.read().strip()

ids = tok.encode(text)

with open("model/tokens.bin", "wb") as f:
    f.write(struct.pack("<i", len(ids)))
    for id in ids:
        f.write(struct.pack("<i", id))

print(f"tokenized {len(ids)} tokens: {ids}")

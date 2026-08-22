import struct
from transformers import GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2-medium")
n_vocab = tok.vocab_size

strings = [tok.decode([i]) for i in range(n_vocab)]

data = b""
offsets = []
for s in strings:
    offsets.append(len(data))
    data += s.encode("utf-8") + b"\x00"

with open("vocab.bin", "wb") as f:
    f.write(struct.pack("<i", n_vocab))
    f.write(struct.pack(f"<{n_vocab}i", *offsets))
    f.write(data)

print(f"wrote vocab.bin: {n_vocab} tokens")

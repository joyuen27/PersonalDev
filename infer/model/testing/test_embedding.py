import struct
import numpy as np

def load_gpt2(path):
    with open(path, "rb") as f:
        magic, n_layer, n_head, n_embd, vocab_size, block_size = struct.unpack("<6i", f.read(24))
        assert magic == 20240326
        wte = np.frombuffer(f.read(vocab_size * n_embd * 4), dtype=np.float32).reshape(vocab_size, n_embd)
        wpe = np.frombuffer(f.read(block_size * n_embd * 4), dtype=np.float32).reshape(block_size, n_embd)
    return wte, wpe, n_embd

def load_tokens(path):
    with open(path, "rb") as f:
        num_tokens, = struct.unpack("<i", f.read(4))
        tokens = list(struct.unpack(f"<{num_tokens}i", f.read(num_tokens * 4)))
    return tokens

wte, wpe, n_embd = load_gpt2("gpt2.bin")
tokens = load_tokens("tokens.bin")

embeddings = np.stack([wte[tok] + wpe[pos] for pos, tok in enumerate(tokens)])

print(f"tokens:     {tokens}")
print(f"embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
print(embeddings)

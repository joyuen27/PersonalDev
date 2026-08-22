import struct
import numpy as np

def load_gpt2(path):
    with open(path, "rb") as f:
        magic, n_layer, n_head, n_embd, vocab_size, block_size = struct.unpack("<6i", f.read(24))
        assert magic == 20240326

        wte   = np.frombuffer(f.read(vocab_size * n_embd * 4), dtype=np.float32).reshape(vocab_size, n_embd)
        wpe   = np.frombuffer(f.read(block_size * n_embd * 4), dtype=np.float32).reshape(block_size, n_embd)
        ln1_w = np.frombuffer(f.read(n_embd * 4),              dtype=np.float32)
        ln1_b = np.frombuffer(f.read(n_embd * 4),              dtype=np.float32)

    return wte, wpe, n_embd, ln1_w, ln1_b

def load_tokens(path):
    with open(path, "rb") as f:
        num_tokens, = struct.unpack("<i", f.read(4))
        tokens = list(struct.unpack(f"<{num_tokens}i", f.read(num_tokens * 4)))
    return tokens

def layer_norm(x, ln_w, ln_b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True, ddof=0)
    return (x - mean) / np.sqrt(var + eps) * ln_w + ln_b


wte, wpe, n_embd, ln1_w, ln1_b = load_gpt2("../gpt2.bin")
tokens = load_tokens("../tokens.bin")

V_embd = np.stack([wte[tok] + wpe[pos] for pos, tok in enumerate(tokens)])
ln_out = layer_norm(V_embd, ln1_w, ln1_b)

c_ln_out = np.frombuffer(open("test_ln_out.bin", "rb").read(), dtype=np.float32).reshape(ln_out.shape)

print(f"tokens: {tokens}")
print(f"layer_norm max err: {np.abs(ln_out - c_ln_out).max():.6e}")

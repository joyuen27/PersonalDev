import struct
import numpy as np

def load_gpt2(path):
    with open(path, "rb") as f:
        magic, n_layer, n_head, n_embd, vocab_size, block_size = struct.unpack("<6i", f.read(24))
        assert magic == 20240326

        f.read(vocab_size * n_embd * 4)  # skip wte
        f.read(block_size * n_embd * 4)  # skip wpe
        f.read(n_embd * 4 * 2)           # skip ln1_w, ln1_b
        f.read(3 * n_embd * n_embd * 4)  # skip attn_w
        f.read(3 * n_embd * 4)           # skip attn_b
        f.read(n_embd * n_embd * 4)      # skip proj_w
        f.read(n_embd * 4)               # skip proj_b
        f.read(n_embd * 4 * 2)           # skip ln2_w, ln2_b

        fc_w   = np.frombuffer(f.read(4 * n_embd * n_embd * 4), dtype=np.float32).reshape(4 * n_embd, n_embd)
        fc_b   = np.frombuffer(f.read(4 * n_embd * 4),          dtype=np.float32)
        proj_w = np.frombuffer(f.read(4 * n_embd * n_embd * 4), dtype=np.float32).reshape(n_embd, 4 * n_embd)
        proj_b = np.frombuffer(f.read(n_embd * 4),              dtype=np.float32)

    return n_embd, fc_w, fc_b, proj_w, proj_b

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.79788456 * (x + 0.044715 * x**3)))

def mlp(ln_out, fc_w, fc_b, proj_w, proj_b):
    fc_out = ln_out @ fc_w.T + fc_b      # [n_tokens, 4*n_embd]
    fc_out = gelu(fc_out)
    return fc_out @ proj_w.T + proj_b    # [n_tokens, n_embd]


n_embd, fc_w, fc_b, proj_w, proj_b = load_gpt2("../gpt2.bin")

ln_out   = np.frombuffer(open("test_mlp_in.bin",  "rb").read(), dtype=np.float32).reshape(-1, n_embd)
mlp_out  = mlp(ln_out, fc_w, fc_b, proj_w, proj_b)

c_mlp_out = np.frombuffer(open("test_mlp_out.bin", "rb").read(), dtype=np.float32).reshape(mlp_out.shape)

print(f"mlp max err: {np.abs(mlp_out - c_mlp_out).max():.6e}")

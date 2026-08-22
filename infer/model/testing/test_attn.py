import struct
import numpy as np

def load_gpt2(path):
    with open(path, "rb") as f:
        magic, n_layer, n_head, n_embd, vocab_size, block_size = struct.unpack("<6i", f.read(24))
        assert magic == 20240326

        wte    = np.frombuffer(f.read(vocab_size * n_embd * 4),  dtype=np.float32).reshape(vocab_size, n_embd)
        wpe    = np.frombuffer(f.read(block_size * n_embd * 4),  dtype=np.float32).reshape(block_size, n_embd)
        _      = f.read(n_embd * 4 * 2)                                            # skip ln1_w, ln1_b
        attn_w = np.frombuffer(f.read(3 * n_embd * n_embd * 4), dtype=np.float32).reshape(3 * n_embd, n_embd)
        attn_b = np.frombuffer(f.read(3 * n_embd * 4),          dtype=np.float32)
        proj_w = np.frombuffer(f.read(n_embd * n_embd * 4),     dtype=np.float32).reshape(n_embd, n_embd)
        proj_b = np.frombuffer(f.read(n_embd * 4),              dtype=np.float32)

    return wte, wpe, n_embd, n_head, attn_w, attn_b, proj_w, proj_b

def load_tokens(path):
    with open(path, "rb") as f:
        num_tokens, = struct.unpack("<i", f.read(4))
        tokens = list(struct.unpack(f"<{num_tokens}i", f.read(num_tokens * 4)))
    return tokens

def layer_norm(x, ln_w, ln_b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True, ddof=0)
    return (x - mean) / np.sqrt(var + eps) * ln_w + ln_b

def attention(ln_out, attn_w, attn_b, proj_w, proj_b, n_head):
    n_tokens, n_embd = ln_out.shape
    head_dim = n_embd // n_head

    QKV = ln_out @ attn_w.T + attn_b                                            # [n_tokens, 3*n_embd]

    Q = QKV[:, :n_embd        ].reshape(n_tokens, n_head, head_dim).transpose(1, 0, 2)
    K = QKV[:, n_embd:2*n_embd].reshape(n_tokens, n_head, head_dim).transpose(1, 0, 2)
    V = QKV[:, 2*n_embd:      ].reshape(n_tokens, n_head, head_dim).transpose(1, 0, 2)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(head_dim)                      # [n_head, n_tokens, n_tokens]
    mask   = np.triu(np.full((n_tokens, n_tokens), -np.inf, dtype=np.float32), k=1)
    scores = scores + mask
    scores = scores - scores.max(axis=-1, keepdims=True)
    scores = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    heads = (scores @ V).transpose(1, 0, 2).reshape(n_tokens, n_embd)          # [n_tokens, n_embd]

    return heads @ proj_w.T + proj_b


wte, wpe, n_embd, n_head, attn_w, attn_b, proj_w, proj_b = load_gpt2("../gpt2.bin")
tokens = load_tokens("../tokens.bin")

V_embd   = np.stack([wte[tok] + wpe[pos] for pos, tok in enumerate(tokens)])
ln_out   = np.frombuffer(open("test_ln_out.bin", "rb").read(), dtype=np.float32).reshape(len(tokens), n_embd)
attn_out = attention(ln_out, attn_w, attn_b, proj_w, proj_b, n_head)

c_attn_out = np.frombuffer(open("test_attn_out.bin", "rb").read(), dtype=np.float32).reshape(attn_out.shape)

print(f"tokens: {tokens}")
print(f"attention max err: {np.abs(attn_out - c_attn_out).max():.6e}")

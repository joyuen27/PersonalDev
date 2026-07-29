from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
sd = model.state_dict()
cfg = model.config

# Get Dimensions
n_layer = cfg.n_layer
n_head = cfg.n_head
n_embd = cfg.n_embd
vocab_size = cfg.vocab_size
block_size = cfg.n_positions

# Mark these to transpose
TRANSPOSE = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}

def get(name):
    # Cast tensor to fp32 
    t = sd[name].float()

    # Check if tensor is marked for transpose
    if any(name.endswith(k) for k in TRANSPOSE):
        # Transpose
        t = t.t().contiguous()
    return t.numpy().tobytes()
 
with open("gpt2.bin", "wb") as f:
    # header: magic, n_layer, n_head, n_embd, vocab_size, block_size (all int32)
    f.write((20240326).to_bytes(4, "little"))
    for v in (n_layer, n_head, n_embd, vocab_size, block_size):
        f.write(v.to_bytes(4, "little"))
 
    # embeddings
    f.write(get("transformer.wte.weight"))   # [vocab_size, n_embd]
    f.write(get("transformer.wpe.weight"))   # [block_size, n_embd]
 
    # per-layer
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        f.write(get(p + "ln_1.weight"))
        f.write(get(p + "ln_1.bias"))
        f.write(get(p + "attn.c_attn.weight"))
        f.write(get(p + "attn.c_attn.bias"))
        f.write(get(p + "attn.c_proj.weight"))
        f.write(get(p + "attn.c_proj.bias"))
        f.write(get(p + "ln_2.weight"))
        f.write(get(p + "ln_2.bias"))
        f.write(get(p + "mlp.c_fc.weight"))
        f.write(get(p + "mlp.c_fc.bias"))
        f.write(get(p + "mlp.c_proj.weight"))
        f.write(get(p + "mlp.c_proj.bias"))
 
    # final layernorm
    f.write(get("transformer.ln_f.weight"))
    f.write(get("transformer.ln_f.bias"))
 
print(f"wrote gpt2.bin: n_layer={n_layer} n_head={n_head} n_embd={n_embd} "
      f"vocab_size={vocab_size} block_size={block_size}")

# Tokenize input
tok = GPT2Tokenizer.from_pretrained("gpt2")

with open("input.txt", "r") as f:
    text = f.read().strip()

ids = tok.encode(text)

with open("tokens.bin", "wb") as f:
    f.write(len(ids).to_bytes(4, "little"))   # header: num_tokens
    for id in ids:
        f.write(id.to_bytes(4, "little"))     # then each token ID, int32
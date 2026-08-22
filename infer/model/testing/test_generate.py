import os
import struct
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- locate files (script may be run from repo root or model/testing) ---
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

def find(name):
    for base in (os.getcwd(), REPO, os.path.join(REPO, "model"), HERE):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)

# --- load the C++ engine's generated token ids ---
with open(find("test_gen_tokens.bin"), "rb") as f:
    prompt_len, total = struct.unpack("<2i", f.read(8))
    c_tokens = np.frombuffer(f.read(total * 4), dtype=np.int32).astype(np.int64)

prompt = c_tokens[:prompt_len].tolist()
max_new = total - prompt_len

# --- PyTorch reference: greedy decode from the same prompt ---
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

ref = list(prompt)
with torch.no_grad():
    for _ in range(max_new):
        logits = model(torch.tensor([ref])).logits  # [1, T, vocab]
        ref.append(int(logits[0, -1].argmax()))

ref = np.array(ref, dtype=np.int64)

# --- compare ---
match = np.array_equal(c_tokens, ref)
print(f"prompt_len={prompt_len}  max_new={max_new}")
print(f"exact token match: {match}")

if not match:
    diff = np.where(c_tokens != ref)[0]
    first = diff[0]
    print(f"tokens match up to step {first - prompt_len} "
          f"({first}/{total} tokens agree)")
    print(f"  first diff @ pos {first}: cpp={c_tokens[first]} pytorch={ref[first]}")

print("\n--- C++ output ---")
print(tok.decode(c_tokens.tolist()))
print("\n--- PyTorch output ---")
print(tok.decode(ref.tolist()))

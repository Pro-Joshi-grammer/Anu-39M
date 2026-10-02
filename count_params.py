"""
Run this FIRST, before touching data or training.
python count_params.py
Adjust GPTConfig in model.py (d_model / n_layer / vocab_size) until total < 50,000,000.
Keep a comfortable margin (e.g. target 40-45M) since the grader includes embeddings + head.
"""
from model import GPT, GPTConfig

cfg = GPTConfig()  # edit model.py defaults, or override here
model = GPT(cfg)

total = model.num_params()
non_emb = model.num_params(non_embedding=True)

print(f"Config: {cfg}")
print(f"Total trainable params (incl. tied embedding/head): {total:,}")
print(f"Non-embedding params: {non_emb:,}")
print(f"Embedding table params (tied, counted once): {total - non_emb:,}")

if total > 50_000_000:
    print("\n⚠️  OVER THE 50M CAP. Reduce d_model, n_layer, d_ff, or vocab_size.")
else:
    print(f"\n✅ Under cap, {50_000_000 - total:,} params of headroom.")

# Save a plain-text record for the README / repo as required by the rules.
with open("param_count.txt", "w") as f:
    f.write(f"Config: {cfg}\n")
    f.write(f"Total trainable params (incl. embeddings + output head): {total:,}\n")

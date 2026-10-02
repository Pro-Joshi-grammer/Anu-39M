"""
Two things required by the rules:
  1. Perplexity on a held-out slice of WikiText-103 (uses the TEST split --
     never seen in training, unlike train.bin which used the TRAIN split).
  2. HellaSwag / ARC-Easy / PIQA / WinoGrande via lm-evaluation-harness.

Run:
  python eval_all.py --ckpt ckpt_final.pt
"""
import argparse
import math
import torch
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from model import GPT, GPTConfig

@torch.no_grad()
def wikitext_perplexity(model, tok, device, block_size=512):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join([r["text"] for r in ds if r["text"].strip()])
    ids = tok.encode(text).ids
    ids = np.array(ids, dtype=np.int64)

    model.eval()
    nlls = []
    stride = block_size
    for i in range(0, len(ids) - block_size, stride):
        chunk = torch.from_numpy(ids[i:i+block_size+1]).unsqueeze(0).to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        nlls.append(loss.item())
        if i // stride >= 2000:  # cap for time -- a few thousand windows is a solid estimate
            break
    mean_nll = sum(nlls) / len(nlls)
    ppl = math.exp(mean_nll)
    print(f"WikiText-103 test perplexity: {ppl:.3f}  (over {len(nlls)} windows)")
    return ppl

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(args.tokenizer)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    ppl = wikitext_perplexity(model, tok, device, block_size=cfg.max_seq_len)

    with open("eval_results.txt", "w") as f:
        f.write(f"WikiText-103 test perplexity: {ppl:.3f}\n")
        f.write("\nFor HellaSwag / ARC-Easy / PIQA / WinoGrande, run lm-evaluation-harness "
                "separately (see run_lm_eval_harness.sh) -- it needs a HF-compatible "
                "model wrapper (hf_wrapper.py) rather than the raw torch checkpoint.\n")

    print("Saved eval_results.txt")

if __name__ == "__main__":
    main()

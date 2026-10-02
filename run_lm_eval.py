"""
Runs HellaSwag, ARC-Easy, PIQA, WinoGrande via lm-evaluation-harness against
our custom (non-HuggingFace) model, using the harness's Python library API
with a minimal custom LM wrapper (loglikelihood-based tasks only, which is
all four of these need).

Install once:
  pip install lm-eval

Run:
  python run_lm_eval.py --ckpt ckpt_final.pt
Takes a while on a T4 -- start this and let it run, it's not interactive.
"""
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPT

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate


class CustomLM(LM):
    def __init__(self, ckpt_path, tokenizer_path, device):
        super().__init__()
        self.device = device
        self.tok = Tokenizer.from_file(tokenizer_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.cfg = ckpt["cfg"]
        self.model = GPT(self.cfg).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.block_size = self.cfg.max_seq_len

    def _encode(self, s):
        return self.tok.encode(s).ids

    @torch.no_grad()
    def loglikelihood(self, requests):
        results = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self._encode(context) if context else [0]
            cont_ids = self._encode(continuation)
            full = (ctx_ids + cont_ids)[-self.block_size - 1:]
            cont_len = min(len(cont_ids), len(full) - 1)
            x = torch.tensor(full[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
            y = torch.tensor(full[1:], dtype=torch.long, device=self.device).unsqueeze(0)
            logits, _ = self.model(x)
            logprobs = F.log_softmax(logits, dim=-1)
            target_logprobs = logprobs[0, -cont_len:, :].gather(
                1, y[0, -cont_len:].unsqueeze(-1)
            ).squeeze(-1)
            greedy = logits[0, -cont_len:, :].argmax(dim=-1)
            is_greedy = bool((greedy == y[0, -cont_len:]).all().item())
            results.append((target_logprobs.sum().item(), is_greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        out = []
        for req in requests:
            (s,) = req.args
            ids = self._encode(s)
            if len(ids) < 2:
                out.append(0.0)
                continue
            x = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)[:, -self.block_size:]
            y = torch.tensor(ids[1:], dtype=torch.long, device=self.device).unsqueeze(0)[:, -self.block_size:]
            logits, _ = self.model(x)
            logprobs = F.log_softmax(logits, dim=-1)
            tgt = logprobs[0].gather(1, y[0].unsqueeze(-1)).squeeze(-1)
            out.append(tgt.sum().item())
        return out

    def generate_until(self, requests):
        # Not needed for HellaSwag/ARC-Easy/PIQA/WinoGrande (all loglikelihood tasks).
        raise NotImplementedError("Not needed for the required benchmark set.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--tasks", default="hellaswag,arc_easy,piqa,winogrande")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = CustomLM(args.ckpt, args.tokenizer, device)

    results = simple_evaluate(model=lm, tasks=args.tasks.split(","), num_fewshot=0)

    print("\n=== RESULTS (copy into README) ===")
    for task, metrics in results["results"].items():
        print(task, metrics)

    with open("lm_eval_results.txt", "w") as f:
        for task, metrics in results["results"].items():
            f.write(f"{task}: {metrics}\n")
    print("\nSaved lm_eval_results.txt")


if __name__ == "__main__":
    main()

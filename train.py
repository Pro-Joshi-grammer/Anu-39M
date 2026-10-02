"""
Training loop for a single GPU (T4, 16GB).
Logs everything you need for the required README reporting section
(hardware, total training time, approx compute).

Usage:
  python train.py                      # full run
  python train.py --max_minutes 180    # hard time cap, auto-stops and saves

Resume anytime: it saves ckpt.pt every --eval_interval steps.
"""
import argparse
import math
import time
import os
import json
import numpy as np
import torch
from model import GPT, GPTConfig

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--max_minutes", type=float, default=None, help="hard wall-clock cap")
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--out", type=str, default="ckpt.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Device: {device} ({gpu_name})")

    train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")

    cfg = GPTConfig(max_seq_len=args.block_size)
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"Model params: {n_params:,}")
    assert n_params <= 50_000_000, "OVER THE PARAM CAP -- fix model.py config before training"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        progress = min(progress, 1.0)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def estimate_val_loss():
        model.eval()
        losses = []
        for _ in range(20):
            x, y = get_batch(val_data, args.block_size, args.batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    t0 = time.time()
    model.train()
    step = 0
    best_val = float("inf")
    log = []

    while step < args.max_steps:
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print("Hit time cap, stopping.")
            break

        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"step {step} | loss {accum_loss:.4f} | lr {lr:.2e} | {elapsed/60:.1f} min elapsed")

        if step % args.eval_interval == 0 and step > 0:
            val_loss = estimate_val_loss()
            ppl = math.exp(min(val_loss, 20))
            print(f"  -> val_loss {val_loss:.4f} | val_ppl {ppl:.2f}")
            log.append({"step": step, "val_loss": val_loss, "val_ppl": ppl,
                        "minutes_elapsed": (time.time() - t0) / 60})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, args.out)

        step += 1

    total_minutes = (time.time() - t0) / 60
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, "ckpt_final.pt")

    report = {
        "hardware": gpu_name,
        "total_training_minutes": round(total_minutes, 1),
        "total_steps": step,
        "tokens_per_step": args.batch_size * args.block_size * args.grad_accum,
        "total_tokens_seen_approx": step * args.batch_size * args.block_size * args.grad_accum,
        "params": n_params,
        "best_val_loss": best_val,
        "val_log": log,
    }
    with open("training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved training_report.json -- copy these numbers into your README's "
          "hardware / training time / compute section (required by the rules).")
    print(json.dumps({k: v for k, v in report.items() if k != "val_log"}, indent=2))

if __name__ == "__main__":
    main()

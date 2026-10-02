import os
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", 200_000_000))
VAL_FRACTION = 0.01
CHUNK_SIZE = 1_000_000

tok = Tokenizer.from_file("tokenizer.json")
eos_id = tok.token_to_id("<eos>")

def encode_to_file(text_iter, token_budget, path):
    total = 0
    buf = []

    with open(path, "wb") as f:
        for text in text_iter:
            if not text or not text.strip():
                continue

            ids = tok.encode(text).ids
            ids.append(eos_id)
            buf.extend(ids)
            total += len(ids)

            if len(buf) >= CHUNK_SIZE:
                n = min(len(buf), token_budget - (total - len(buf)))
                np.asarray(buf[:n], dtype=np.uint16).tofile(f)
                buf = buf[n:]

            if total >= token_budget:
                break

        if buf and total > token_budget:
            n = len(buf) - (total - token_budget)
            buf = buf[:n]

        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)

    return total


def wikitext_iter():
    ds = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="train"
    )
    for row in ds:
        yield row["text"]


def tinystories_iter():
    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True
    )
    for row in ds:
        yield row["text"]


def fineweb_iter():
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    for row in ds:
        yield row["text"]


targets = {
    "wikitext": int(TOKEN_BUDGET * 0.40),
    "tinystories": int(TOKEN_BUDGET * 0.30),
    "fineweb": int(TOKEN_BUDGET * 0.30),
}

print("Encoding WikiText-103...")
wt = encode_to_file(
    tqdm(wikitext_iter()),
    targets["wikitext"],
    "wikitext_part.bin"
)
print(f"WikiText tokens: {wt:,}")

print("Encoding TinyStories...")
ts = encode_to_file(
    tqdm(tinystories_iter()),
    targets["tinystories"],
    "tinystories_part.bin"
)
print(f"TinyStories tokens: {ts:,}")

print("Encoding FineWeb-Edu...")
fw = encode_to_file(
    tqdm(fineweb_iter()),
    targets["fineweb"],
    "fineweb_part.bin"
)
print(f"FineWeb-Edu tokens: {fw:,}")

total = wt + ts + fw
print(f"Total tokens: {total:,}")

train_tokens = int(total * (1 - VAL_FRACTION))

with open("train.bin", "wb") as train_f, \
     open("val.bin", "wb") as val_f:

    written = 0

    for path in [
        "wikitext_part.bin",
        "tinystories_part.bin",
        "fineweb_part.bin",
    ]:
        data = np.memmap(path, dtype=np.uint16, mode="r")

        pos = 0
        while pos < len(data):
            n = min(CHUNK_SIZE, len(data) - pos)
            chunk = data[pos:pos+n]

            if written < train_tokens:
                train_n = min(n, train_tokens - written)
                chunk[:train_n].tofile(train_f)

                if train_n < n:
                    chunk[train_n:].tofile(val_f)
            else:
                chunk.tofile(val_f)

            written += n
            pos += n

        del data

print("Dataset creation complete.")

train_count = os.path.getsize("train.bin") // 2
val_count = os.path.getsize("val.bin") // 2

print(f"train.bin: {train_count:,} tokens")
print(f"val.bin:   {val_count:,} tokens")
print(f"Total:     {train_count + val_count:,} tokens")

for path in [
    "wikitext_part.bin",
    "tinystories_part.bin",
    "fineweb_part.bin",
]:
    os.remove(path)

print("Temporary files removed.")

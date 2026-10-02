"""
Trains a 16K-vocab BPE tokenizer on a quick sample of the training mix.
Fast (~2-5 min). Run once, before data_prep.py.
"""
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 16000
SAMPLE_DOCS = 40000  # enough to fit a good BPE vocab, kept small for speed

def text_iterator():
    # WikiText-103 train split (matches the eval domain)
    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    n = 0
    for row in wt:
        if row["text"].strip():
            yield row["text"]
            n += 1
            if n >= SAMPLE_DOCS // 2:
                break
    # TinyStories for simple narrative coverage
    ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    n = 0
    for row in ts:
        yield row["text"]
        n += 1
        if n >= SAMPLE_DOCS // 2:
            break

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    show_progress=True,
)

tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
tokenizer.save("tokenizer.json")
print("Saved tokenizer.json, vocab size:", tokenizer.get_vocab_size())

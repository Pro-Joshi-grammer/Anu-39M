# Anu-39M — GIBC V2 Track 01

A 39.66M-parameter decoder-only Transformer language model trained from scratch under the 50M parameter limit.

## Model

|                |                                                              |
| -------------- | ------------------------------------------------------------ |
| Architecture   | Decoder-only Transformer, RoPE, tied input/output embeddings |
| Parameters     | **39,660,032**                                               |
| Layers         | 10                                                           |
| d_model        | 512                                                          |
| Heads          | 8                                                            |
| Vocab size     | 16,000 custom BPE                                            |
| Context length | 512                                                          |

Parameter count can be verified with `count_params.py`. The output is saved in `param_count.txt`.

## Training Data

- **WikiText-103** — training split.
- **TinyStories** — general narrative text.
- **FineWeb-Edu** — educational and general-domain text.

The final training corpus contained approximately **393.2M training tokens**.

The model and tokenizer were trained from scratch. No pretrained model weights were used.

## Training

- **Hardware:** AWS g4dn.xlarge, NVIDIA Tesla T4 16GB, 4 vCPU
- **Training time:** **165 minutes**
- **Training steps:** **6,000**
- **Tokens seen:** **393,216,000**
- **Precision:** FP16 mixed precision
- **Parameters:** **39,660,032**
- **Approximate compute:** **9.38 × 10¹⁶ FLOPs (~26.1 GPU-hours at 1 TFLOP/s-equivalent accounting)**

The best validation checkpoint was obtained at step 5,750 with a validation perplexity of **38.03**.

## Evaluation

Evaluation was performed using the best validation checkpoint (`ckpt.pt`).

| Benchmark                    |      Score |
| ---------------------------- | ---------: |
| WikiText-103 test perplexity | **38.608** |
| HellaSwag                    | **26.78%** |
| ARC-Easy                     | **33.25%** |
| PIQA                         | **56.15%** |
| WinoGrande                   | **49.88%** |

HellaSwag, ARC-Easy, PIQA and WinoGrande were evaluated 0-shot using `lm-evaluation-harness`.

## Design Choices

- **RoPE** was used instead of learned positional embeddings.
- **Tied embeddings** reduce the parameter count and leave more of the budget for Transformer layers.
- A **16K BPE vocabulary** keeps the embedding and output layers relatively small.
- The model uses **10 Transformer layers with 512-dimensional hidden states** while remaining below the 50M parameter limit.
- The training corpus combines WikiText-103, TinyStories and FineWeb-Edu to provide a mixture of general and educational text.

## How to Run

```bash
pip install -r requirements.txt

python tokenizer_train.py
python data_prep.py
python count_params.py

python train.py --max_minutes 240

python eval_all.py --ckpt ckpt_final.pt
python run_lm_eval.py --ckpt ckpt_final.pt

Built With

PyTorch, Hugging Face Tokenizers, Hugging Face Datasets and lm-evaluation-harness.

AI Tool Disclosure

Claude and ChatGPT were used during development to assist with code, debugging, implementation decisions and documentation. The resulting code was run and tested on the training environment.

Limitations

This is a small language model trained under a strict parameter and compute budget. Its 512-token context length and 39.66M parameter size limit its ability to handle long-context and complex language tasks
```

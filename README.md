# slm_tiny.stories
# SLM Project — Small Language Model from Scratch

A minimal, from-scratch implementation of a GPT-style decoder-only Transformer, trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The project walks through the full pipeline: data loading, tokenization, model architecture, training, evaluation, and text generation — implemented in pure PyTorch with no high-level model libraries.

## Overview

This notebook builds a small language model (SLM) end-to-end:

1. **Data** — Loads the TinyStories dataset (short, simple children's stories) via Hugging Face `datasets`.
2. **Tokenization** — Encodes text using the GPT-2 BPE tokenizer (`tiktoken`).
3. **Model** — Implements a decoder-only Transformer (multi-head self-attention, feed-forward blocks, residual connections, layer norm) from scratch using `torch.nn`.
4. **Training** — Trains with AdamW, cosine learning-rate decay, gradient clipping, and periodic train/val loss evaluation.
5. **Generation** — Samples text from the trained model using top-k sampling with temperature control.
6. **Checkpointing** — Saves and reloads the trained model for inference.

## Model Architecture

A GPT-style decoder-only Transformer built from these components:

- `Head` — single causal self-attention head (scaled dot-product attention with causal masking)
- `MultiHeadAttention` — multiple attention heads run in parallel and combined
- `FeedForward` — position-wise MLP (expand → ReLU → project)
- `Block` — one Transformer block (attention + feed-forward, each with residual connection and pre-LayerNorm)
- `GPT` — token + positional embeddings, stack of `Block`s, final LayerNorm, and a language-modeling head

## Default Configuration

| Parameter | Value |
|---|---|
| `num_stories` | 8,000 (training subset) |
| `block_size` | 128 (context length) |
| `vocab_size` | 50,257 (GPT-2 BPE) |
| `n_layer` | 4 |
| `n_head` | 8 |
| `n_embd` | 256 |
| `dropout` | 0.1 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 (cosine decay) |
| `max_steps` | 2,000 |
| `eval_interval` | 200 |
| `eval_iters` | 50 |
| `seed` | 1337 |

All hyperparameters live in a single `config` dictionary near the top of the notebook and are threaded through every downstream cell, so they can be adjusted in one place.

## Requirements

- Python 3.9+
- PyTorch (with CUDA support recommended, but CPU works for small runs)
- `datasets`
- `tiktoken`
- `numpy`
- `matplotlib`
- `tqdm`

Install dependencies:

```bash
pip install torch datasets tiktoken numpy matplotlib tqdm
```

(The notebook's first cell also runs `pip install -q datasets tiktoken` directly.)

## Project Structure

```
SLM_project.ipynb        # main notebook: data → model → training → generation
data/
  train.bin               # tokenized training data (uint16 binary)
  val.bin                 # tokenized validation data (uint16 binary)
outputs/
  loss_curve.png          # training vs. validation loss plot
  model.pt                # saved model checkpoint (weights + config + final losses)
```

`data/` and `outputs/` are created automatically by the notebook if they don't already exist.

## Usage

1. **Open and run the notebook** (`SLM_project.ipynb`) top to bottom, e.g. in Jupyter or Google Colab (GPU runtime recommended for faster training).
2. **Environment setup** — installs dependencies and detects CUDA availability.
3. **Data preparation** — downloads TinyStories, tokenizes it with GPT-2 BPE, and writes `data/train.bin` / `data/val.bin`.
4. **Build the model** — instantiates the `GPT` class with the values in `config`.
5. **Train** — runs the training loop for `max_steps`, logging train/val loss every `eval_interval` steps and plotting the resulting loss curve.
6. **Generate text** — prompts the trained model (e.g. `"Once upon a time"`) and samples continuations at different temperatures.
7. **Save / reload** — writes a checkpoint to `outputs/model.pt` and demonstrates reloading it for inference.

### Example: generating text after training

```python
prompt = "Once upon a time"
prompt_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=config["device"])
generated = model.generate(prompt_ids, max_new_tokens=150, temperature=0.8, top_k=50)
print(enc.decode(generated[0].tolist()))
```

### Example: loading a saved checkpoint

```python
checkpoint = torch.load("outputs/model.pt", map_location=device)
model = GPT(checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Notes

- The model is intentionally small (a few million parameters) so it can be trained quickly on a single GPU (or CPU, more slowly) as a learning exercise in Transformer internals rather than a production-scale LLM.
- Generation quality reflects the small model size, limited training subset (8,000 stories), and short training run (2,000 steps) — increasing `num_stories`, `max_steps`, `n_layer`, `n_embd`, and `n_head` in `config` will improve output quality at the cost of longer training time.
- The random seed (`1337`) is fixed for reproducibility across `torch`, `numpy`, and CUDA.

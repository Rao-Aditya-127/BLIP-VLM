# BLIP-VLM: Q-Former Pre-training (BLIP-2 Stage 1)

A from-scratch implementation of the Q-Former component from [BLIP-2](https://arxiv.org/abs/2301.12597), trained with ITC + ITM losses on the Conceptual Captions dataset.

## Overview

The Q-Former bridges a frozen vision encoder (ViT) and a language model by learning a fixed set of query tokens that extract the most task-relevant visual features. This repo implements **Stage 1** pre-training:

- **ITC** (Image-Text Contrastive): aligns image and text representations using per-query max similarity and a learnable temperature.
- **ITM** (Image-Text Matching): binary classification with hard negative mining to distinguish matched vs. mismatched image-text pairs.

## Project Structure

```
BLIP-VLM/
├── q_former_train.py          # Main training script
├── networks/
│   ├── q_former.py            # QFormer model definition
│   └── blip2_trainer.py       # Blip2QFormerTrainer (ITC + ITM losses)
└── datasets/
    ├── cc_dataloader.py        # Conceptual Captions dataloader
    └── lm_dataloader.py
```

## How It Works

1. **Vision encoding**: Images are passed through `google/vit-base-patch16-224` to get patch embeddings `[B, 197, 768]`.
2. **Q-Former**: 32 learnable query tokens attend to visual features via cross-attention (every 2 layers) and to text via self-attention, using `distilbert-base-uncased` as the transformer backbone with a separate per-layer query FFN.
3. **ITC loss**: Query outputs and text CLS token are projected to 256-d, normalized, and matched via contrastive loss with label smoothing.
4. **ITM loss**: Hard negatives are mined from the ITC similarity matrix; the Q-Former runs in bidirectional multi-modal mode and a linear head classifies match/no-match.

## Setup

```bash
pip install torch transformers pillow pyarrow tqdm
```

Place the Conceptual Captions dataset under `dataset/`:
```
dataset/
├── cc_images/          # images downloaded via img2dataset
└── conceptual-captions-200k.parquet
```

## Training

```bash
python q_former_train.py
```

Key hyperparameters (set at the top of `q_former_train.py`):

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Batch size | 8 |
| Epochs | 10 |
| Embed dim (ITC) | 256 |
| Query tokens | 32 |
| Temperature init | 0.07 |

Checkpoints are saved to `models/trained_qformer_v2/`:
- `best/` — lowest combined ITC + ITM test loss
- `latest/` — periodic checkpoint every 20 steps

## Architecture Notes

- DistilBERT weights are fine-tuned at `lr * 0.1`; new modules (cross-attention blocks, query tokens, projection heads) use the full lr.
- Biases and LayerNorm parameters are excluded from weight decay (following original BLIP-2).
- Temperature is clamped to `[0.001, 0.5]` after each optimizer step.

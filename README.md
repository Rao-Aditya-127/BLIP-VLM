# BLIP-VLM

A from-scratch implementation of a Vision-Language Model inspired by [BLIP-2](https://arxiv.org/abs/2301.12597), trained on MS COCO for image captioning and visual question answering.

---

## Architecture

Three frozen/trainable components connected in sequence:

```
ViT (frozen) → Q-Former (trainable) → Adapter (trainable) → SmolLM-135M + LoRA (trainable)
```

| Component | Model | Params | Status |
|---|---|---|---|
| Vision Encoder | google/vit-base-patch16-224 | 86M | Frozen |
| Q-Former | DistilBERT backbone + cross-attention | ~66M | Trainable |
| Adapter | 2-layer MLP | ~1M | Trainable |
| LLM | HuggingFaceTB/SmolLM-135M-Instruct + LoRA (r=64) | 19.5M / 154M | LoRA only |

Training follows BLIP-2's two-stage approach:
- **Stage 1** — Q-Former pre-training with ITC + ITM losses
- **Stage 2** — LLM fine-tuning with cross-entropy on caption tokens

---

## Project Structure

```
BLIP-VLM/
├── networks/
│   ├── q_former.py            # Q-Former model
│   ├── blip2_trainer.py       # Stage 1 trainer (ITC + ITM losses)
│   └── lm_to_vlm.py           # Stage 2 model (ViT + QFormer + Adapter + LLM)
├── datasets_loader/
│   ├── mscoco_dataloader.py   # Stage 1 dataloader
│   └── lm_dataloader.py       # Stage 2 dataloader
├── mscoco_train.py            # Stage 1 training script
├── lm_train.py                # Stage 2 training script
├── lm_train_resume.py         # Resume Stage 2 from checkpoint
├── Experimentation/
│   ├── qformer_evaluation.ipynb   # Stage 1 evaluation
│   └── vlm_evaluation.ipynb       # Stage 2 evaluation (caption gen, VQA)
├── models_trained/
│   └── trained_qformer_mscoco/best/qformer/
└── models_trained_vlm/
    └── vlm_peft/best/
```

---

## Setup

```bash
pip install torch torchvision transformers peft accelerate
pip install pillow pyarrow tqdm jupyter matplotlib scikit-learn
```

---

## Dataset

**MS COCO train2017** — 118,287 images with 5 captions each (591,753 total pairs).

Downloaded via [img2dataset](https://github.com/rom1504/img2dataset). Expected directory layout:

```
dataset/
└── mscoco_images/
    ├── 00000/          # image shards
    ├── 00000.parquet   # metadata (key, caption, status)
    ├── 00001/
    ├── 00001.parquet
    └── ...
```

```bash
img2dataset --url_list mscoco_train2017.parquet \
            --output_folder dataset/mscoco_images \
            --processes_count 16 \
            --thread_count 64 \
            --image_size 224
```

---

## Training

### Stage 1 — Q-Former Pre-training

```bash
python mscoco_train.py
```

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Learning rate | 1e-4 |
| Epochs | 10 |
| Optimizer | AdamW |
| Mixed precision | bfloat16 |
| Hardware | A100 80GB |

Checkpoints saved to `models/trained_qformer_mscoco/` (`best/`, `latest/`).

### Stage 2 — LLM Fine-tuning

```bash
python lm_train.py
```

| Hyperparameter | Value |
|---|---|
| Batch size | 128 |
| lr (Q-Former) | 2e-4 |
| lr (Adapter + LoRA) | 1e-3 |
| Gradient accumulation | 4 |
| Warmup steps | 200 |
| Epochs | 5 |
| Mixed precision | bfloat16 |
| Hardware | A100 80GB |

To resume from a saved checkpoint:

```bash
python lm_train_resume.py
```

Checkpoints saved to `models/vlm_peft/` (`best/`, `latest/`, `final/`).

---

## Evaluation

### Stage 1

```bash
jupyter notebook Experimentation/qformer_evaluation.ipynb
```

- Image → Text and Text → Image retrieval
- Recall@K (quantitative)
- ITM scoring
- t-SNE embedding visualisation

### Stage 2

```bash
jupyter notebook Experimentation/vlm_evaluation.ipynb
```

- Caption generation vs ground truth
- Multi-prompt comparison
- Visual question answering
- Generation parameter sweep

### Quick Inference

```python
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer
from networks.lm_to_vlm import LM_2_VLM
import torch

model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = LM_2_VLM(model_name=model_name, pad_token_id=tokenizer.pad_token_id)
model.load_checkpoint("models_trained_vlm/vlm_peft/best")
model.eval()

if not torch.cuda.is_available():
    model.float()

processor    = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
image        = Image.open("your_image.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

prompt_text = tokenizer.apply_chat_template(
    [{"role": "system", "content": "Answer the user's question truthfully"},
     {"role": "user",   "content": "Describe this image in detail."}],
    tokenize=False, add_generation_prompt=False,
)
prefix_ids = torch.tensor([tokenizer.encode(prompt_text)])

with torch.no_grad():
    output_ids = model.generate(img=pixel_values, prefix_ids=prefix_ids)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Results

### Stage 1 — Q-Former

Trained for **10 epochs (~11 hours)** on a single A100 80GB (GPU 0).

| Metric | Value |
|---|---|
| ITC loss | 0.7498 |
| ITM loss  | 0.2084 |
| Total loss | 0.9582 |
| Image→Text R@1 | ~28.7% |
| Image→Text R@5 | ~89.3% |
| Image→Text R@10 | ~98.7% |

> **Note:** The A100's 80GB VRAM was heavily underutilised at `batch_size=32` (~10% utilisation). Increasing to `batch_size=512` or higher would reduce training time to ~2 hours for the same 10 epochs and significantly improve ITC/ITM quality through harder negative mining.

### Stage 2 — VLM

Trained for **2 epochs** on a single A100 80GB, limited by compute budget.

| Metric | Value |
|---|---|
| Test loss  | 1.5161 |
| Epochs completed | 2 of 5 (target) |

The model produces captions that are reasonably relevant to image content, demonstrating that the visual pipeline (ViT → Q-Former → Adapter → LLM) is functioning correctly. However, with only 2 epochs of training — and the adapter starting from random initialisation — the outputs lack the consistency and descriptive detail that more training would provide. A full 5-epoch run is expected to close this gap significantly.

**VQA limitation:** The model performs poorly on specific visual questions (e.g. "How many people are in this photo?"). The root cause is a training data mismatch — every training example used a generic captioning prompt ("Describe this image", "What do you see?") with a description as the target. The model never saw specific question-answer pairs, so it learned a single behaviour: regardless of the question asked, output a description of the image.

To fix this, the Stage 2 dataloader needs to incorporate the MSCOCO VQA annotations alongside the caption data:

```
v2_OpenEnded_mscoco_train2014_questions.json   # 443k questions
v2_mscoco_train2014_annotations.json           # 443k answers
```

Mixed training on both tasks would teach the model to answer specific questions with precise responses ("Red", "3 people", "Daytime") rather than always defaulting to a caption.

---

## References

- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)

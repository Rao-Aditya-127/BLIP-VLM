"""
Q-Former training with ITC + ITM losses (BLIP-2 Stage 1 style).

Compared to q_former_train.py (ITC only), this adds:
- Per-query max similarity (instead of mean-pooled CLIP loss)
- Learnable temperature
- ITM loss with hard negative mining
- Projection heads (vision_proj, text_proj)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from networks.q_former import QFormer
from networks.blip2_trainer import Blip2QFormerTrainer
import torch
import torch.optim as optim
from torch.optim import AdamW
from transformers import DistilBertModel
from datasets.cc_dataloader import get_dataloaders
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Device: {device}")

# --- Model setup ---
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
qformer = QFormer(bert)
model = Blip2QFormerTrainer(qformer, embed_dim=256)
model.to(device)

model_id = "trained_qformer_v2"
lr = 1e-4
batch_size = 8

train_loader, test_loader = get_dataloaders(batch_size=batch_size)

# --- Optimizer with grouped learning rates and weight decay ---
# AdamW matches original BLIP-2; weight_decay=0.01 applies only to non-bias/non-norm params
optimizer = AdamW(model.get_optimizer_params(lr, weight_decay=0.01))


def run_inference(limit_batches=20):
    model.eval()
    losses_itc, losses_itm = [], []
    with torch.no_grad():
        for i, (img, txt) in enumerate(test_loader):
            if i >= limit_batches:
                break

            img = img.to(device)
            if isinstance(txt, dict):
                txt = {k: v.to(device) for k, v in txt.items()}

            result = model(
                image_embeds=img,
                text_input_ids=txt["input_ids"],
                text_attention_mask=txt["attention_mask"],
            )
            losses_itc.append(result["loss_itc"].item())
            losses_itm.append(result["loss_itm"].item())
    model.train()

    if not losses_itc:
        return float("inf"), float("inf")
    return np.mean(losses_itc), np.mean(losses_itm)


# --- Training loop ---
steps = 0
log_train_loss_every = 5
run_inference_every = 10
save_checkpoint_every = 20
best_test_loss = np.inf

for epoch in range(10):
    train_losses = {"itc": [], "itm": [], "total": []}
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for img, txt in pbar:
        steps += 1

        img = img.to(device)
        if isinstance(txt, dict):
            txt = {k: v.to(device) for k, v in txt.items()}

        result = model(
            image_embeds=img,
            text_input_ids=txt["input_ids"],
            text_attention_mask=txt["attention_mask"],
        )
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Clamp temperature after optimizer step to prevent instability
        with torch.no_grad():
            model.temp.clamp_(0.001, 0.5)

        train_losses["itc"].append(result["loss_itc"].item())
        train_losses["itm"].append(result["loss_itm"].item())
        train_losses["total"].append(loss.item())
        pbar.set_postfix(
            itc=f"{result['loss_itc'].item():.3f}",
            itm=f"{result['loss_itm'].item():.3f}",
            temp=f"{model.temp.item():.4f}",
        )

        if steps % log_train_loss_every == 0:
            tqdm.write(
                f"Epoch: {epoch+1}, Steps: {steps}, "
                f"ITC: {np.mean(train_losses['itc']):.4f}, "
                f"ITM: {np.mean(train_losses['itm']):.4f}, "
                f"Total: {np.mean(train_losses['total']):.4f}, "
                f"Temp: {model.temp.item():.4f}"
            )
            train_losses = {"itc": [], "itm": [], "total": []}

        if steps % run_inference_every == 0:
            test_itc, test_itm = run_inference()
            test_total = test_itc + test_itm
            tqdm.write(
                f"Steps: {steps}, Test ITC: {test_itc:.4f}, "
                f"Test ITM: {test_itm:.4f}, Test Total: {test_total:.4f}"
            )

            if test_total < best_test_loss:
                best_model_dir = f"models/{model_id}/best"
                model.save_pretrained(best_model_dir)
                tqdm.write(f"New best model saved in {best_model_dir}")
                best_test_loss = test_total

        if steps % save_checkpoint_every == 0:
            model.save_pretrained(f"models/{model_id}/latest")
            tqdm.write(f"Checkpoint saved at step {steps}")

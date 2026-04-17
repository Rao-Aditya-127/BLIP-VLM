"""
Temporary resume script — loads checkpoint from models/vlm_peft/best and
continues training for 1 epoch at batch_size=128.
Delete this file after use.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets_loader.lm_dataloader import get_dataloader
import torch
import torch.optim as optim
from tqdm import tqdm
from networks.lm_to_vlm import LM_2_VLM
import numpy as np
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator

if __name__ == "__main__":
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="logs",
    )

    model_name   = "HuggingFaceTB/SmolLM-135M-Instruct"
    checkpoint   = "models/vlm_peft/best"   # checkpoint from previous run
    model_id     = "vlm_peft"

    # --- Data ---
    train_loader, test_loader = get_dataloader(batch_size=128, tokenizer_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # --- Model (initialise then load checkpoint weights) ---
    model = LM_2_VLM(
        model_name=model_name,
        qformer_model_path="models_trained/trained_qformer_mscoco/best/qformer",
        pad_token_id=pad_token_id,
    )
    model.load_checkpoint(checkpoint)
    accelerator.print(f"Resumed from checkpoint: {checkpoint}")

    # --- Optimizer ---
    lr_slow = 2e-4
    lr_fast = 1e-3

    qformer_params = model.qformer.get_grouped_params()
    optimizer = optim.AdamW(
        [
            {"params": qformer_params["default"],          "lr": lr_slow},
            {"params": qformer_params["cross_blocks"],     "lr": lr_slow},
            {"params": qformer_params["query_embeddings"], "lr": lr_slow},
            {"params": model.adapter.parameters(),         "lr": lr_fast},
            {"params": filter(lambda p: p.requires_grad, model.llm.parameters()),
             "lr": lr_fast},
        ]
    )

    # --- Schedule: 1 epoch ---
    epochs       = 1
    warmup_steps = 100
    max_grad_norm = 1.0
    log_every    = 20
    save_every   = 100

    total_steps = len(train_loader) * epochs // accelerator.gradient_accumulation_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    # --- Inference helper ---
    def run_inference(limit_batches=20):
        model.eval()
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i >= limit_batches:
                    break
                with accelerator.autocast():
                    output = model(data["image"], data["prefix"], data["assistant_prompt"])
                loss = accelerator.gather(output.loss).mean()
                losses.append(loss.item())
        model.train()
        return np.mean(losses) if losses else float("inf")

    # --- Training ---
    step = 0
    best_test_loss = float("inf")
    model.train()

    accelerator.print(f"Starting resumed training — {len(train_loader)} steps/epoch")

    pbar = tqdm(
        train_loader,
        desc="Resumed epoch",
        disable=not accelerator.is_local_main_process,
    )

    for data in pbar:
        with accelerator.accumulate(model):
            with accelerator.autocast():
                output = model(data["image"], data["prefix"], data["assistant_prompt"])
                loss = output.loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_local_main_process:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        step += 1

        if step % log_every == 0 and accelerator.is_local_main_process:
            test_loss = run_inference()
            accelerator.print(
                f"Step {step} | Train Loss: {loss.item():.4f} | "
                f"Test Loss: {test_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                accelerator.unwrap_model(model).save_checkpoint(f"models/{model_id}/best")
                accelerator.print(f"✓ New best model saved! Loss: {best_test_loss:.4f}")

        if step % save_every == 0 and accelerator.is_local_main_process:
            accelerator.unwrap_model(model).save_checkpoint(f"models/{model_id}/latest")

    # Save final
    if accelerator.is_local_main_process:
        accelerator.unwrap_model(model).save_checkpoint(f"models/{model_id}/final")
        accelerator.print("Resumed training complete.")

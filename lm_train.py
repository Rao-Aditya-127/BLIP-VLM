from datasets_loader.lm_dataloader import get_dataloader
import torch
import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm
from networks.lm_to_vlm import LM_2_VLM
import numpy as np
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
if __name__ == "__main__":
    # --- Initialize Accelerator ---
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",  # Use bfloat16 mixed precision
        log_with="tensorboard",
        project_dir="logs",
    )

    model_id = "vlm_peft"
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

    train_loader, test_loader = get_dataloader(batch_size=128, tokenizer_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    model = LM_2_VLM(
        model_name=model_name,
        qformer_model_path="models_trained/trained_qformer_mscoco/best/qformer",
        pad_token_id=pad_token_id,
    )

    # --- Optimizer Setup ---
    lr_slow = 2e-4    # scaled from 1e-4 × √(128/32)
    lr_fast = 1e-3    # scaled from 5e-4 × √(128/32)

    qformer_params = model.qformer.get_grouped_params()
    optimizer = optim.AdamW(
        [
            {"params": qformer_params["default"], "lr": lr_slow},
            {"params": qformer_params["cross_blocks"], "lr": lr_slow},
            {"params": qformer_params["query_embeddings"], "lr": lr_slow},
            {"params": model.adapter.parameters(), "lr": lr_fast},
            {
                "params": filter(lambda p: p.requires_grad, model.llm.parameters()),
                "lr": lr_fast,
            },
        ]
    )

    # --- Training Configuration ---
    epochs = 5
    log_every = 20
    save_every = 100
    warmup_steps = 200  # fewer steps/epoch at batch_size=256
    max_grad_norm = 1.0  # Gradient clipping threshold

    # Calculate total training steps
    total_steps = len(train_loader) * epochs // accelerator.gradient_accumulation_steps

    # --- Cosine LR Scheduler ---
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- Prepare with Accelerator ---
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    step = 0
    best_test_loss = float("inf")

    def run_inference(model, test_loader, limit_batches=20):
        model.eval()
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i >= limit_batches:
                    break

                img = data["image"]
                prefix = data["prefix"]
                assistant = data["assistant_prompt"]

                with accelerator.autocast():
                    output = model(img, prefix, assistant)

                # Gather losses from all processes if using distributed training
                loss = accelerator.gather(output.loss).mean()
                losses.append(loss.item())

        model.train()

        if not losses:
            return float("inf")
        return np.mean(losses)

    model.train()

    accelerator.print("Starting training...")
    accelerator.print(f"Total training steps: {total_steps}")
    accelerator.print(
        f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )

    for epoch in range(epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for data in pbar:
            with accelerator.accumulate(model):
                img = data["image"]
                prefix = data["prefix"]
                assistant = data["assistant_prompt"]

                with accelerator.autocast():
                    output = model(img, prefix, assistant)
                    loss = output.loss

                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Only log on main process
            if accelerator.is_local_main_process:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

            step += 1

            if step % log_every == 0 and accelerator.is_local_main_process:
                test_loss = run_inference(model, test_loader)
                accelerator.print(
                    f"Step {step} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    # Unwrap model before saving
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_checkpoint(f"models/{model_id}/best")
                    accelerator.print(
                        f"✓ New best model saved! Loss: {best_test_loss:.4f}"
                    )

            if step % save_every == 0 and accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_checkpoint(f"models/{model_id}/latest")

    # Save final model
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_checkpoint(f"models/{model_id}/final")
        accelerator.print("Training complete.")

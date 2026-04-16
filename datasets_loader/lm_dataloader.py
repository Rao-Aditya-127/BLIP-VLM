from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import ViTImageProcessor, AutoTokenizer
import random

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


@dataclass(frozen=True)
class MSCOCOExample:
    image_path: Path
    caption: str


class LMDataset(Dataset):
    """
    Torch-style Dataset for MS COCO images for LLM training.

    Returns pixel_values [3, 224, 224] + tokenized prefix/assistant prompts.
    ViT inference is excluded — the model runs it batched on GPU in forward().
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset",
        vit_model: str = "google/vit-base-patch16-224",
        tokenizer: Optional[str] = None,
    ) -> None:
        self.images_root = Path(dataset_root, "mscoco_images")
        # Processor is CPU-only: safe to use inside DataLoader workers
        self.vit_processor = ViTImageProcessor.from_pretrained(vit_model)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self._examples: list[MSCOCOExample] = self._build_index()

        self.prompts = [
            "Tell me about this image:",
            "Describe this picture.",
            "What do you see in this image?",
            "Provide a description of the photo.",
            "Can you explain what is shown in this image?",
            "What is in this picture?",
            "Describe the contents of this image.",
            "Give me a summary of what's shown here.",
            "What can you see here?",
            "Explain the visual content of this image.",
            "Describe this image in detail.",
            "What's happening in this photo?",
        ]

    def _build_index(self) -> list[MSCOCOExample]:
        out: list[MSCOCOExample] = []
        for parquet_file in sorted(self.images_root.glob("*.parquet")):
            shard_name = parquet_file.stem  # e.g. "00000"
            shard_dir = self.images_root / shard_name
            if not shard_dir.is_dir():
                continue
            table = pq.read_table(parquet_file, columns=["key", "caption", "status"])
            keys = table["key"].to_pylist()
            captions = table["caption"].to_pylist()
            statuses = table["status"].to_pylist()
            for key, caption, status in zip(keys, captions, statuses):
                if status != "success":
                    continue
                if not caption:
                    continue
                jpg_file = shard_dir / f"{key}.jpg"
                if not jpg_file.exists():
                    continue
                out.append(
                    MSCOCOExample(
                        image_path=jpg_file,
                        caption=str(caption).strip(),
                    )
                )
        return out

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]

        with Image.open(ex.image_path) as im:
            image = im.convert("RGB").copy()

        # CPU-only preprocessing — no GPU, safe in DataLoader workers
        pixel_values = self.vit_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze(0)  # [3, 224, 224]

        random_prompt = random.choice(self.prompts)
        user_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Answer the user's question truthfully"},
                {"role": "user", "content": random_prompt},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        user_prompt = torch.tensor(
            [self.tokenizer.encode(user_text)]
        ).to(device)  # [1, seq_len]

        assistant_text = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ex.caption}],
            tokenize=False,
            add_generation_prompt=False,
        )
        assistant_prompt = torch.tensor(
            [self.tokenizer.encode(assistant_text)]
        )  # [1, seq_len]

        # Trim any tokens after the last EOS
        eos_positions = (assistant_prompt[0] == self.tokenizer.eos_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(eos_positions) > 0:
            last_eos_idx = eos_positions[-1].item()
            assistant_prompt = assistant_prompt[:, : last_eos_idx + 1]

        assistant_prompt = assistant_prompt.to(device)

        return {
            "image_filename": str(ex.image_path),
            "caption": ex.caption,
            "image": pixel_values,
            "prefix": user_prompt,
            "assistant_prompt": assistant_prompt,
        }


class LMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [item["image"] for item in batch]

        # Ensure 1D for padding
        prefixes = [
            item["prefix"].squeeze(0) if item["prefix"].ndim == 2 else item["prefix"]
            for item in batch
        ]
        assistant_prompts = [
            (
                item["assistant_prompt"].squeeze(0)
                if item["assistant_prompt"].ndim == 2
                else item["assistant_prompt"]
            )
            for item in batch
        ]

        images = torch.stack(images)  # [B, 3, 224, 224]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                raise ValueError(
                    "Tokenizer must have a pad_token_id or eos_token_id set."
                )

        # Left-pad prefixes
        max_prefix_len = max([p.size(0) for p in prefixes])
        prefixes_padded = torch.full(
            (len(prefixes), max_prefix_len), pad_id, dtype=torch.long
        )
        for i, p in enumerate(prefixes):
            prefixes_padded[i, -len(p) :] = p

        # Right-pad assistant prompts
        assistant_prompts_padded = pad_sequence(
            assistant_prompts, batch_first=True, padding_value=pad_id
        )

        return {
            "image": images.to(device),
            "prefix": prefixes_padded.to(device),
            "assistant_prompt": assistant_prompts_padded.to(device),
        }


def get_dataloader(
    batch_size=4,
    split_ratio=0.9,
    seed=42,
    tokenizer_name="HuggingFaceTB/SmolLM-135M-Instruct",
):
    dataset = LMDataset(tokenizer=tokenizer_name)

    if dataset.tokenizer.pad_token is None:
        dataset.tokenizer.pad_token = dataset.tokenizer.eos_token

    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    collator = LMCollator(dataset.tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloader(batch_size=4)
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    for d in train_loader:
        print("Image shape:          ", d["image"].shape)           # [4, 3, 224, 224]
        print("Prefix shape:         ", d["prefix"].shape)
        print("Assistant prompt shape:", d["assistant_prompt"].shape)
        break

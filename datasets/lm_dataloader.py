from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
import pyarrow.parquet as pq
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import ViTModel, ViTImageProcessor, AutoTokenizer
import numpy as np
import random

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


@dataclass(frozen=True)
class CCExample:
    image_path: Path
    caption: str


class LMDataset(Dataset):
    """
    Torch-style Dataset for Conceptual Captions images downloaded via img2dataset.

    Returns by default: (PIL.Image, caption)
    Set `return_image_path=True` to return (Path, caption) instead.
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset",
        vit_model: str = "google/vit-base-patch16-224",
        tokenizer: Optional[str] = None,
        return_image_path: bool = False,
    ) -> None:
        self.images_root = Path(dataset_root, "cc_images")
        self.index_parquet = Path(dataset_root, "conceptual-captions-200k.parquet")

        self.vit_processor = ViTImageProcessor.from_pretrained(vit_model)
        self.vit_model = ViTModel.from_pretrained(vit_model)
        self.vit_model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.return_image_path = return_image_path
        self._examples: list[CCExample] = self._build_index()

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

    def _build_image_paths(self):
        # Loop recursively 2 directories down and find all jpg files
        jpg_files = {}
        for subdir1 in self.images_root.iterdir():
            if not subdir1.is_dir():
                continue
            for file in subdir1.iterdir():
                if file.is_file() and file.suffix.lower() == ".jpg":
                    if file.name.startswith("."):
                        continue
                    file_idx = int(file.name.split(".")[0])
                    jpg_files[file_idx] = os.path.join(
                        self.images_root, subdir1.name, file.name
                    )
        return jpg_files

    def _load_caption_index(self) -> Dict[str, str]:
        table = pq.read_table(self.index_parquet, columns=["url", "caption"])
        urls = table["url"].to_pylist()
        caps = table["caption"].to_pylist()

        url_to_caption: Dict[str, str] = {}
        for u, c in zip(urls, caps):
            if u is None:
                continue
            if c is None:
                continue
            url_to_caption[str(u)] = str(c)
        return url_to_caption

    def _build_index(self) -> list[CCExample]:
        image_files = self._build_image_paths()
        url_to_caption = self._load_caption_index()

        table = pq.read_table(self.index_parquet, columns=["url", "caption"])
        captions = table["caption"].to_pylist()
        out: list[CCExample] = []
        for idx, caption in enumerate(captions):
            if idx in image_files:
                out.append(
                    CCExample(
                        image_path=image_files[idx],
                        caption=caption,
                    )
                )
        return out

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any] | Dict[str, Any]:
        ex = self._examples[idx]
        caption: Any = ex.caption

        with Image.open(ex.image_path) as im:
            image = im.convert("RGB").copy()

        with torch.no_grad():
            image = self.vit_processor(images=image, return_tensors="pt").to(
                self.vit_model.device
            )
            image = self.vit_model(**image).last_hidden_state
        # [1, num_patches, hidden_dim] -> [num_patches, hidden_dim]
        image = image.squeeze(0)

        random_prompt = random.choice(self.prompts)
        user_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Answer the user's question truthfully"},
                {"role": "user", "content": random_prompt},
            ],
            return_tensors="pt",
        ).to(device)

        assistant_prompt = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": caption}],
            return_tensors="pt",
            add_generation_prompt=False,
        )

        # Ensure sequence ends with EOS token (trim any trailing tokens like newlines)
        # Find the last occurrence of EOS token and truncate after it
        eos_positions = (assistant_prompt[0] == self.tokenizer.eos_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(eos_positions) > 0:
            last_eos_idx = eos_positions[-1].item()
            assistant_prompt = assistant_prompt[:, : last_eos_idx + 1]

        assistant_prompt = assistant_prompt.to(device)

        return {
            "image_filename": ex.image_path,
            "caption": caption,
            "image": image,
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

        images = torch.stack(images)

        # Determine padding value
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                raise ValueError(
                    "Tokenizer must have a pad_token_id or eos_token_id set."
                )

        # Left Pad Prefixes manually
        max_prefix_len = max([p.size(0) for p in prefixes])
        prefixes_padded = torch.full(
            (len(prefixes), max_prefix_len), pad_id, dtype=torch.long
        )
        for i, p in enumerate(prefixes):
            prefixes_padded[i, -len(p) :] = p

        # Pad sequences (right padding) for assistant prompts
        assistant_prompts_padded = pad_sequence(
            assistant_prompts, batch_first=True, padding_value=pad_id
        )

        return {
            "image": images.to(device),
            "prefix": prefixes_padded.to(device),
            "assistant_prompt": assistant_prompts_padded.to(device),
        }


def get_dataset(split_ratio=0.9, seed=42, tokenizer_name="Qwen/Qwen3-0.6B"):

    dataset = LMDataset(tokenizer=tokenizer_name)
    # Ensure pad_token is set for Qwen if using it directly, though Collator handles fallback to EOS
    if dataset.tokenizer.pad_token is None:
        dataset.tokenizer.pad_token = dataset.tokenizer.eos_token

    # Split dataset into train and test
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset.dataset, test_dataset.dataset


def get_dataloader(
    batch_size=4, split_ratio=0.9, seed=42, tokenizer_name="Qwen/Qwen3-0.6B"
):

    dataset = LMDataset(tokenizer=tokenizer_name)
    # Ensure pad_token is set for Qwen if using it directly, though Collator handles fallback to EOS
    if dataset.tokenizer.pad_token is None:
        dataset.tokenizer.pad_token = dataset.tokenizer.eos_token

    # Split dataset into train and test
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

    train_loader, test_loader = get_dataloader()
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    for d in train_loader:
        print("Image shape:", d["image"].shape)
        print("Prefix shape:", d["prefix"].shape)
        print("Assistant prompt shape:", d["assistant_prompt"].shape)
        print(d["prefix"])
        print(d["assistant_prompt"])

        break

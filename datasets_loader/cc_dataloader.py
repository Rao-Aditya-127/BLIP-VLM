from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
import pyarrow.parquet as pq
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from transformers import ViTModel, ViTImageProcessor, AutoTokenizer
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


@dataclass(frozen=True)
class CCExample:
    image_path: Path
    caption: str


class CCImageCaptionDataset(Dataset):
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
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = None

        self.return_image_path = return_image_path
        self._examples: list[CCExample] = self._build_index()

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
        # Remove batch dimension (will be added back in collate_fn)
        image = image.squeeze(
            0
        )  # [1, num_patches, hidden_dim] -> [num_patches, hidden_dim]

        # Return raw caption string - tokenization will happen in collate_fn for proper batching
        return image, caption


def collate_fn(
    batch: List[Tuple[Any, Any]], tokenizer: Optional[AutoTokenizer] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    images, captions = zip(*batch)

    image_tensors = torch.stack(images, dim=0).to(device)
    if tokenizer is not None:
        # Tokenize with padding
        tokenized = tokenizer(
            list(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        return image_tensors, tokenized
    else:
        # No tokenizer - return captions as-is (list of strings)
        return image_tensors, list(captions)


def get_dataloaders(
    vit_model="google/vit-base-patch16-224",
    tokenizer="distilbert/distilbert-base-uncased",
    batch_size=16,
    split_ratio=0.9,
    seed=42,
):

    dataset = CCImageCaptionDataset(vit_model=vit_model, tokenizer=tokenizer)

    # Split dataset into train and test
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    # Create collate function with tokenizer from dataset
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=dataset.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_with_tokenizer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_with_tokenizer,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    for batch in train_loader:
        images, captions = batch
        print(f"Batch - images shape: {images.shape}")
        if isinstance(captions, dict):
            print(captions["input_ids"])
            print(f"Batch - captions input_ids shape: {captions['input_ids'].shape}")
            print(
                f"Batch - captions attention_mask shape: {captions['attention_mask'].shape}"
            )
        else:
            print(f"Batch - captions: {len(captions)} items")
        break

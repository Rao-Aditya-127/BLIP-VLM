from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import partial
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from transformers import ViTImageProcessor, AutoTokenizer


@dataclass(frozen=True)
class MSCOCOExample:
    image_path: Path
    caption: str


class MSCOCOImageCaptionDataset(Dataset):
    """
    Torch-style Dataset for MS COCO images downloaded via img2dataset.

    Returns (pixel_values [3, 224, 224], caption_str) per sample.
    ViT inference is intentionally excluded — run it batched on GPU in the
    training loop so the GPU stays busy instead of waiting for per-sample CPU work.
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

        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = None

        self._examples: list[MSCOCOExample] = self._build_index()

    def _build_index(self) -> list[MSCOCOExample]:
        out: list[MSCOCOExample] = []
        # Read per-shard parquet files (60 reads) instead of 286k individual JSON sidecars
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        ex = self._examples[idx]
        with Image.open(ex.image_path) as im:
            image = im.convert("RGB").copy()
        # Only preprocessing (resize + normalize) — no GPU, safe in workers
        pixel_values = self.vit_processor(images=image, return_tensors="pt").pixel_values
        return pixel_values.squeeze(0), ex.caption  # [3, 224, 224], str


def collate_fn(
    batch: List[Tuple[torch.Tensor, str]],
    tokenizer: Optional[AutoTokenizer] = None,
) -> Tuple[torch.Tensor, Dict]:
    pixel_values, captions = zip(*batch)
    pixel_batch = torch.stack(pixel_values, dim=0)  # [B, 3, 224, 224]
    if tokenizer is not None:
        tokenized = tokenizer(
            list(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return pixel_batch, tokenized
    return pixel_batch, list(captions)


def get_dataloaders(
    vit_model: str = "google/vit-base-patch16-224",
    tokenizer: str = "distilbert/distilbert-base-uncased",
    batch_size: int = 32,
    num_workers: int = 4,
    split_ratio: float = 0.9,
    seed: int = 42,
):
    dataset = MSCOCOImageCaptionDataset(vit_model=vit_model, tokenizer=tokenizer)

    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    collate = partial(collate_fn, tokenizer=dataset.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    print(f"Train: {len(train_loader)} batches | Test: {len(test_loader)} batches")
    pixel_batch, captions = next(iter(train_loader))
    print(f"pixel_values shape: {pixel_batch.shape}")   # [4, 3, 224, 224]
    if isinstance(captions, dict):
        print(f"input_ids shape:    {captions['input_ids'].shape}")

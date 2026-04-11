from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
import os
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from transformers import ViTModel, ViTImageProcessor, AutoTokenizer

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


@dataclass(frozen=True)
class MSCOCOExample:
    image_path: Path
    caption: str


class MSCOCOImageCaptionDataset(Dataset):
    """
    Torch-style Dataset for MS COCO images downloaded via img2dataset.

    Reads captions directly from the .json sidecars written by img2dataset,
    filtering to only images with status == "success".

    Returns by default: (image_embeds, caption_str)
    Set `return_image_path=True` to return (Path, caption) instead.
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset",
        vit_model: str = "google/vit-base-patch16-224",
        tokenizer: Optional[str] = None,
        return_image_path: bool = False,
    ) -> None:
        self.images_root = Path(dataset_root, "mscoco_images")

        self.vit_processor = ViTImageProcessor.from_pretrained(vit_model)
        self.vit_model = ViTModel.from_pretrained(vit_model)
        self.vit_model.to(device)

        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = None

        self.return_image_path = return_image_path
        self._examples: list[MSCOCOExample] = self._build_index()

    def _build_index(self) -> list[MSCOCOExample]:
        out: list[MSCOCOExample] = []
        # Each shard has a corresponding .parquet at images_root/XXXXX.parquet
        # Reading 60 parquet files is vastly faster than opening 286k .json sidecars
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

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        ex = self._examples[idx]

        if self.return_image_path:
            return ex.image_path, ex.caption

        with Image.open(ex.image_path) as im:
            image = im.convert("RGB").copy()

        with torch.no_grad():
            image = self.vit_processor(images=image, return_tensors="pt").to(
                self.vit_model.device
            )
            image = self.vit_model(**image).last_hidden_state
        # Remove batch dimension (will be added back in collate_fn)
        image = image.squeeze(0)  # [num_patches, hidden_dim]

        return image, ex.caption


def collate_fn(
    batch: List[Tuple[Any, Any]], tokenizer: Optional[AutoTokenizer] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    images, captions = zip(*batch)

    image_tensors = torch.stack(images, dim=0).to(device)
    if tokenizer is not None:
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
        return image_tensors, list(captions)


def get_dataloaders(
    vit_model="google/vit-base-patch16-224",
    tokenizer="distilbert/distilbert-base-uncased",
    batch_size=16,
    split_ratio=0.9,
    seed=42,
):
    dataset = MSCOCOImageCaptionDataset(vit_model=vit_model, tokenizer=tokenizer)

    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

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
            print(f"Batch - captions input_ids shape: {captions['input_ids'].shape}")
            print(
                f"Batch - captions attention_mask shape: {captions['attention_mask'].shape}"
            )
        else:
            print(f"Batch - captions: {len(captions)} items")
        break

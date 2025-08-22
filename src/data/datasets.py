"""Datasets for supervised and semi-supervised training."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T
from torch.utils.data import Dataset

from .augmentations import weak_transform, strong_transform


class SkinPairDataset(Dataset):
    """Return weak/strong augmentations and mask for (image, mask) pairs."""

    def __init__(self, pairs: List[Tuple[str, str]], size: Tuple[int, int] = (256, 256)):
        self.pairs = pairs
        self.w_t = weak_transform()
        self.s_t = strong_transform()
        self.transform_mask = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_p, msk_p = self.pairs[idx]
        img = Image.open(img_p).convert("RGB")
        mask = self.transform_mask(Image.open(msk_p).convert("L"))
        mask = (mask > 0.5).float()  # binarize
        return self.w_t(img), self.s_t(img), mask


class UnlabeledDataset(Dataset):
    """Return weak/strong augmentations for images without masks."""

    def __init__(self, img_paths: List[str]):
        self.imgs = img_paths
        self.w_t = weak_transform()
        self.s_t = strong_transform()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = Image.open(self.imgs[idx]).convert("RGB")
        return self.w_t(img), self.s_t(img)


class CocoSegDataset(Dataset):
    """Dataset that builds masks on-the-fly from COCO annotations."""

    def __init__(
        self,
        img_dir: str | Path,
        ann_file: str | Path,
        ids: Optional[Sequence[int]] = None,
        size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_file))
        self.ids = list(ids) if ids is not None else list(self.coco.imgs.keys())
        self.w_t = weak_transform()
        self.s_t = strong_transform()
        self.transform_mask = T.Compose(
            [T.Resize(size, interpolation=T.InterpolationMode.NEAREST), T.ToTensor()]
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        mask = Image.fromarray(mask * 255)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()
        return self.w_t(img), self.s_t(img), mask


__all__ = ["SkinPairDataset", "UnlabeledDataset", "CocoSegDataset"]


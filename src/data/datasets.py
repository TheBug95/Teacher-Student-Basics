"""Datasets for supervised and semi-supervised training."""

from typing import List, Tuple

from PIL import Image
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


__all__ = ["SkinPairDataset", "UnlabeledDataset"]


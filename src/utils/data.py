"""Utility helpers for data loading."""

from pathlib import Path
from typing import List, Tuple


def find_pairs(img_dir: Path, mask_dir: Path) -> List[Tuple[str, str]]:
    """Match images and masks using common filename patterns."""
    pairs: List[Tuple[str, str]] = []
    for m in mask_dir.glob("*.png"):
        img = img_dir / m.name.replace("_segmentation", "").replace(".png", ".jpg")
        if img.exists():
            pairs.append((str(img), str(m)))
    return pairs


__all__ = ["find_pairs"]

from pathlib import Path
from typing import List, Tuple

from .metrics import dice_score  # re-export for backward compatibility

def find_pairs(img_dir: Path, mask_dir: Path) -> List[Tuple[str, str]]:
    pairs = []
    for m in mask_dir.glob("*.png"):
        img = img_dir / m.name.replace("_segmentation", "").replace(".png", ".jpg")
        if img.exists():
            pairs.append((str(img), str(m)))
    return pairs

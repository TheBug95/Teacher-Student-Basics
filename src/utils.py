from pathlib import Path
from typing import List, Tuple
import torch

def dice_score(logits, targets, thr=.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice  = (2*inter + eps) / (union + eps)
    return dice.mean()

def find_pairs(img_dir: Path, mask_dir: Path) -> List[Tuple[str,str]]:
    pairs = []
    for m in mask_dir.glob("*.png"):
        img = img_dir / m.name.replace("_segmentation", "").replace(".png", ".jpg")
        if img.exists():
            pairs.append((str(img), str(m)))
    return pairs

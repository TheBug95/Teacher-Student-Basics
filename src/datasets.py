from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class SkinPairDataset(Dataset):
    """Lee (imagen, máscara) y aplica una única transformación débil."""
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        size: Tuple[int, int] = (256, 256)
    ):
        self.pairs = pairs
        self.transform_img = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_p, msk_p = self.pairs[idx]
        img  = self.transform_img(Image.open(img_p).convert("RGB"))
        mask = self.transform_mask(Image.open(msk_p).convert("L"))
        mask = (mask > .5).float()          # binariza
        return img, mask

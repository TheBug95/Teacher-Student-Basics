import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

_SIZE = (256, 256)

def weak_transform():
    return T.Compose([
        T.Resize(_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1, 0.1),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],
                    [0.229,0.224,0.225])
    ])

class CutOut:
    """1 agujero negro cuadrado."""
    def __init__(self, length=64): self.length = length
    def __call__(self, img):
        import torchvision.transforms.functional as F
        import torch, random
        _, h, w = img.shape
        y = random.randint(0, h)
        x = random.randint(0, w)
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        img[:, y1:y2, x1:x2] = 0.
        return img

def strong_transform():
    return T.Compose([
        T.Resize(_SIZE, interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(15)], p=0.5),
        T.ToTensor(),
        CutOut(64),
        T.Normalize([0.485,0.456,0.406],
                    [0.229,0.224,0.225])
    ])

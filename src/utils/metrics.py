import torch
from torchmetrics.functional import structural_similarity_index_measure


def dice_score(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Computa la métrica Dice entre predicciones y máscaras."""
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Intersection over Union para máscaras binarias."""
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def ssim_score(logits: torch.Tensor, targets: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Structural Similarity Index Measure."""
    probs = torch.sigmoid(logits)
    return structural_similarity_index_measure(probs, targets, data_range=data_range)


__all__ = ["dice_score", "iou_score", "ssim_score"]

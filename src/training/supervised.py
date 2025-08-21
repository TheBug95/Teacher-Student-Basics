"""Supervised training script."""

import argparse
from pathlib import Path

import torch
import tqdm

from src.data import SkinPairDataset
from src.models import get_model
from src.utils import dice_score, iou_score, ssim_score, find_pairs


def run(args: argparse.Namespace) -> None:
    img_dir = Path(args.images)
    mask_dir = Path(args.masks)

    pairs = find_pairs(img_dir, mask_dir)
    split = int(len(pairs) * 0.8)
    train_ds = SkinPairDataset(pairs[:split])
    val_ds = SkinPairDataset(pairs[split:])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.bs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        tl, td, ti, ts = 0.0, 0.0, 0.0, 0.0
        for x, y in tqdm.tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            td += dice_score(logits, y).item() * x.size(0)
            ti += iou_score(logits, y).item() * x.size(0)
            ts += ssim_score(logits, y).item() * x.size(0)
        tl /= len(train_ds)
        td /= len(train_ds)
        ti /= len(train_ds)
        ts /= len(train_ds)

        model.eval()
        vl, vd, vi, vs = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vl += crit(logits, y).item() * x.size(0)
                vd += dice_score(logits, y).item() * x.size(0)
                vi += iou_score(logits, y).item() * x.size(0)
                vs += ssim_score(logits, y).item() * x.size(0)
        vl /= len(val_ds)
        vd /= len(val_ds)
        vi /= len(val_ds)
        vs /= len(val_ds)

        print(
            f"[{epoch+1}] Train Loss {tl:.4f} Dice {td:.4f} IoU {ti:.4f} SSIM {ts:.4f} | "
            f"Val Loss {vl:.4f} Dice {vd:.4f} IoU {vi:.4f} SSIM {vs:.4f}"
        )

        torch.save(model.state_dict(), f"{args.model}_supervised.pth")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="carpeta con imágenes JPG")
    parser.add_argument("--masks", required=True, help="carpeta con máscaras PNG")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--model",
        default="sam2",
        choices=["sam2", "medsam2", "mobilesam", "unet"],
        help="modelo de segmentación",
    )
    run(parser.parse_args())


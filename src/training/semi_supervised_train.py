"""Teacher-student semi-supervised training script."""

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm

from src.data import CocoSegDataset, UnlabeledDataset
from src.models import get_model
from src.utils import dice_score, iou_score, ssim_score, update_ema


def run(args: argparse.Namespace) -> None:
    # --- 1. Datos --------------------------------------------------------
    data_dir = Path(args.train)
    full_ds = CocoSegDataset(data_dir, data_dir / "_annotations.coco.json")
    valid_split = args.valid_split

    ids = full_ds.ids
    cut = int(len(ids) * 0.8)
    labeled_ids = ids[:cut]
    unlabeled_ids = ids[cut:]

    num_valid = int(len(labeled_ids) * valid_split)
    train_ids = labeled_ids[:-num_valid]
    val_ids = labeled_ids[-num_valid:]

    l_train = CocoSegDataset(data_dir, data_dir / "_annotations.coco.json", ids=train_ids)
    l_val = CocoSegDataset(data_dir, data_dir / "_annotations.coco.json", ids=val_ids)

    unlabeled_paths = [
        str(data_dir / full_ds.coco.loadImgs(i)[0]["file_name"]) for i in unlabeled_ids
    ]
    u_train = UnlabeledDataset(unlabeled_paths)

    l_dl = torch.utils.data.DataLoader(l_train, batch_size=args.bs, shuffle=True)
    u_dl = torch.utils.data.DataLoader(u_train, batch_size=args.bs, shuffle=True)
    val_dl = torch.utils.data.DataLoader(l_val, batch_size=args.bs)

    # --- 2. Modelos ------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = get_model(args.model).to(device)
    teacher = get_model(args.model).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    # --- 3. Entrenamiento -----------------------------------------------
    u_iter = iter(itertools.cycle(u_dl))
    for epoch in range(args.epochs):
        student.train()
        t_loss = t_dice = t_iou = t_ssim = 0.0
        for weak_l, strong_l, mask_l in tqdm.tqdm(
            l_dl, desc=f"Ep {epoch+1}/{args.epochs}"
        ):
            weak_l, strong_l, mask_l = (
                x.to(device) for x in (weak_l, strong_l, mask_l)
            )

            # supervised loss
            logit_sup = student(weak_l)
            loss_sup = bce(logit_sup, mask_l)

            # unsupervised consistency
            weak_u, strong_u = next(u_iter)
            weak_u, strong_u = weak_u.to(device), strong_u.to(device)

            with torch.no_grad():
                pseudo = torch.sigmoid(teacher(weak_u))

            conf_mask = (pseudo > args.tau).float()
            if conf_mask.sum() > 0:
                logit_u = student(strong_u)
                loss_cons = F.binary_cross_entropy_with_logits(
                    logit_u, (pseudo > 0.5).float(), weight=conf_mask
                )
            else:
                loss_cons = torch.tensor(0.0, device=device)

            loss = loss_sup + args.lmbda * loss_cons
            opt.zero_grad()
            loss.backward()
            opt.step()

            update_ema(student, teacher, alpha=args.ema)

            t_loss += loss.item() * weak_l.size(0)
            t_dice += dice_score(logit_sup, mask_l).item() * weak_l.size(0)
            t_iou += iou_score(logit_sup, mask_l).item() * weak_l.size(0)
            t_ssim += ssim_score(logit_sup, mask_l).item() * weak_l.size(0)

        # --- 4. Validación ------------------------------------------------
        student.eval()
        v_dice = v_iou = v_ssim = 0.0
        with torch.no_grad():
            for weak, _, mask in val_dl:
                weak, mask = weak.to(device), mask.to(device)
                logits = student(weak)
                v_dice += dice_score(logits, mask).item() * weak.size(0)
                v_iou += iou_score(logits, mask).item() * weak.size(0)
                v_ssim += ssim_score(logits, mask).item() * weak.size(0)

        v_dice /= len(l_val)
        v_iou /= len(l_val)
        v_ssim /= len(l_val)

        print(
            f"Epoch {epoch+1} | Train Dice {t_dice/len(l_train):.4f} "
            f"IoU {t_iou/len(l_train):.4f} SSIM {t_ssim/len(l_train):.4f} | "
            f"Val Dice {v_dice:.4f} IoU {v_iou:.4f} SSIM {v_ssim:.4f}"
        )

        torch.save(student.state_dict(), f"{args.model}_semi.pth")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tau", type=float, default=0.7, help="umbral confianza")
    p.add_argument("--lmbda", type=float, default=1.0, help="peso consistencia")
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--valid_split", type=float, default=0.2)
    p.add_argument(
        "--model",
        default="sam2",
        choices=["sam2", "medsam2", "mobilesam", "unet"],
        help="modelo de segmentación",
    )
    run(p.parse_args())


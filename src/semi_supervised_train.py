import argparse, torch, tqdm, itertools
from pathlib import Path
from src.datasets import SkinPairDataset, UnlabeledDataset
from src.metrics import dice_score, iou_score, ssim_score
from src.utils import find_pairs
from src.ema import update_ema
from src.models   import get_model

import torch.nn.functional as F

def run(args):
    # --- 1. Datos ----------------------------------------------------------
    # --- 1. Datos ----------------------------------------------------------
    img_dir  = Path(args.images)
    mask_dir = Path(args.masks)

    # 0. Parámetro para tamaño de validación
    valid_split = getattr(args, "valid_split", 0.2)

    # 1. Hallar todos los pares (ruta_img, ruta_mask)
    pairs = find_pairs(img_dir, mask_dir)          # len(pairs) = N

    # 2. Dividir labeled / unlabeled (80 % / 20 % ejemplo)
    cut = int(len(pairs) * 0.8)

    # ── Bloque labeled (que SÍ usa la máscara) ──────────────────────────
    labeled_pairs   = pairs[:cut]                  # 80 %
    num_valid       = int(len(labeled_pairs) * valid_split)

    train_pairs     = labeled_pairs[:-num_valid]   # 80 % * 80 %
    val_pairs       = labeled_pairs[-num_valid:]   # 80 % * 20 %

    l_train = SkinPairDataset(train_pairs)         # weak, strong, mask
    l_val   = SkinPairDataset(val_pairs)

    # ── Bloque unlabeled (que IGNORA la máscara) ────────────────────────
    unlabeled_paths = [img_path for img_path, _ in pairs[cut:]]   # solo la ruta IMG
    u_train = UnlabeledDataset(unlabeled_paths)    # weak, strong

    l_dl = torch.utils.data.DataLoader(l_train, batch_size=args.bs, shuffle=True)
    u_dl = torch.utils.data.DataLoader(u_train, batch_size=args.bs, shuffle=True)
    val_dl = torch.utils.data.DataLoader(l_val, batch_size=args.bs)

    # --- 2. Modelos --------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = get_model(args.model).to(device)
    teacher = get_model(args.model).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    opt  = torch.optim.Adam(student.parameters(), lr=args.lr)
    bce  = torch.nn.BCEWithLogitsLoss()

    # --- 3. Entrenamiento --------------------------------------------------
    u_iter = iter(itertools.cycle(u_dl))   # iterador infinito
    for epoch in range(args.epochs):
        student.train(); t_loss, t_dice, t_iou, t_ssim = 0, 0, 0, 0
        for weak_l, strong_l, mask_l in tqdm.tqdm(l_dl, desc=f"Ep {epoch+1}/{args.epochs}"):
            weak_l, strong_l, mask_l = (x.to(device) for x in (weak_l, strong_l, mask_l))

            # -------- supervisado --------
            logit_sup = student(weak_l)
            loss_sup  = bce(logit_sup, mask_l)

            # -------- no supervisado -----
            weak_u, strong_u = next(u_iter)
            weak_u, strong_u = weak_u.to(device), strong_u.to(device)

            with torch.no_grad():
                pseudo = torch.sigmoid(teacher(weak_u))

            conf_mask = (pseudo > args.tau).float()
            if conf_mask.sum() > 0:
                logit_u    = student(strong_u)
                loss_cons  = F.binary_cross_entropy_with_logits(
                                logit_u, (pseudo > 0.5).float(),
                                weight=conf_mask)
            else:
                loss_cons = torch.tensor(0., device=device)

            loss = loss_sup + args.lmbda * loss_cons
            opt.zero_grad(); loss.backward(); opt.step()

            update_ema(student, teacher, alpha=args.ema)

            t_loss += loss.item() * weak_l.size(0)
            t_dice += dice_score(logit_sup, mask_l).item() * weak_l.size(0)
            t_iou += iou_score(logit_sup, mask_l).item() * weak_l.size(0)
            t_ssim += ssim_score(logit_sup, mask_l).item() * weak_l.size(0)

        # --- 4. Validación --------------------------------------------------
        student.eval(); v_dice, v_iou, v_ssim = 0, 0, 0
        with torch.no_grad():
            for weak, _, mask in val_dl:
                weak, mask = weak.to(device), mask.to(device)

                logits = student(weak)
                v_dice += dice_score(logits, mask).item() * weak.size(0)
                v_iou += iou_score(logits, mask).item() * weak.size(0)
                v_ssim += ssim_score(logits, mask).item() * weak.size(0)
        v_dice /= len(l_val); v_iou /= len(l_val); v_ssim /= len(l_val)

        print(
            f"Epoch {epoch+1} | Train Dice {t_dice/len(l_train):.4f} IoU {t_iou/len(l_train):.4f} "
            f"SSIM {t_ssim/len(l_train):.4f} | Val Dice {v_dice:.4f} IoU {v_iou:.4f} SSIM {v_ssim:.4f}"
      
        torch.save(student.state_dict(), f"{args.model}_semi.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--masks",  required=True)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tau", type=float, default=0.7, help="umbral confianza")
    p.add_argument("--lmbda", type=float, default=1.0, help="peso consistencia")
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument(
        "--model",
        default="sam2",
        choices=["sam2", "medsam2", "mobilesam", "unet"],
        help="modelo de segmentación",
    )
    run(p.parse_args())

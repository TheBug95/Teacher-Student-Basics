import argparse, torch, tqdm, itertools
from pathlib import Path
from src.datasets import SkinPairDataset, UnlabeledDataset
from src.model     import UNet
from src.utils     import dice_score, find_pairs
from src.ema       import update_ema
import torch.nn.functional as F

def run(args):
    # --- 1. Datos ----------------------------------------------------------
    img_dir  = Path(args.images)
    mask_dir = Path(args.masks)
    valid_split = args.valid_split if hasattr(args, 'valid_split') else 0.2

    pairs = find_pairs(img_dir, mask_dir)
    split = int(len(pairs)*0.8)
    l_train = pairs[:split]
    num_valid = int(len(l_train) * valid_split)
    
    l_train     = SkinPairDataset(l_train[:-num_valid])
    l_val       = SkinPairDataset(l_train[-num_valid:])
    unl_train   = SkinPairDataset(pairs[split:])

    u_train  = UnlabeledDataset(unl_train)

    l_dl = torch.utils.data.DataLoader(l_train, batch_size=args.bs, shuffle=True)
    u_dl = torch.utils.data.DataLoader(u_train, batch_size=args.bs, shuffle=True)
    val_dl = torch.utils.data.DataLoader(l_val, batch_size=args.bs)

    # --- 2. Modelos --------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = UNet().to(device)
    teacher = UNet().to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    opt  = torch.optim.Adam(student.parameters(), lr=args.lr)
    bce  = torch.nn.BCEWithLogitsLoss()

    # --- 3. Entrenamiento --------------------------------------------------
    u_iter = iter(itertools.cycle(u_dl))   # iterador infinito
    for epoch in range(args.epochs):
        student.train(); t_loss, t_dice = 0, 0
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

            t_loss += loss.item()*weak_l.size(0)
            t_dice += dice_score(logit_sup, mask_l).item()*weak_l.size(0)

        # --- 4. Validación --------------------------------------------------
        student.eval(); v_dice = 0
        with torch.no_grad():
            for weak, _, mask in val_dl:
                weak, mask = weak.to(device), mask.to(device)
                v_dice += dice_score(student(weak), mask).item()*weak.size(0)
        v_dice /= len(l_val)

        print(f"Epoch {epoch+1} | Train Dice {t_dice/len(l_train):.4f} | Val Dice {v_dice:.4f}")
        torch.save(student.state_dict(), "unet_semi.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--masks",  required=True)
    p.add_argument("--unlabeled", required=True)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tau", type=float, default=0.7, help="umbral confianza")
    p.add_argument("--lmbda", type=float, default=1.0, help="peso consistencia")
    p.add_argument("--ema", type=float, default=0.99)
    run(p.parse_args())

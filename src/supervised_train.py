import argparse, torch, tqdm
from pathlib import Path

from src.datasets import SkinPairDataset
from src.model     import UNet
from src.utils     import dice_score, find_pairs

def run(args):
    img_dir  = Path(args.images)
    mask_dir = Path(args.masks)

    pairs = find_pairs(img_dir, mask_dir)
    split = int(len(pairs)*0.8)
    train_ds = SkinPairDataset(pairs[:split])
    val_ds   = SkinPairDataset(pairs[split:])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=args.bs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = UNet().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit   = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train(); tl, td = 0, 0
        for x, y in tqdm.tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            tl += loss.item()*x.size(0)
            td += dice_score(logits, y).item()*x.size(0)
        tl /= len(train_ds); td /= len(train_ds)

        model.eval(); vl, vd = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vl += crit(logits, y).item()*x.size(0)
                vd += dice_score(logits, y).item()*x.size(0)
        vl /= len(val_ds); vd /= len(val_ds)

        print(f"[{epoch+1}] Train Loss {tl:.4f} Dice {td:.4f} | Val Loss {vl:.4f} Dice {vd:.4f}")

        torch.save(model.state_dict(), "unet_supervised.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="carpeta con imágenes JPG")
    p.add_argument("--masks",  required=True, help="carpeta con máscaras PNG")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    run(p.parse_args())

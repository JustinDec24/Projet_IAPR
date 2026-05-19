"""Entraîne le classifieur de cartes UNO sur le dataset synthétique.

Usage:
    python scripts/train_classifier.py [--epochs 20] [--batch-size 128]

Sortie : outputs/models/classifier.pt (state_dict + métadonnées).
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, NUM_CLASSES  # noqa: E402
from src.dataset import UnoTemplateDataset  # noqa: E402
from src.model import UnoCNN, count_parameters  # noqa: E402

TEMPLATES_DIR = ROOT / "data" / "card_templates"
BG_PATCHES_DIR = ROOT / "data" / "bg_patches"
REAL_CROPS_DIR = ROOT / "data" / "real_crops"
CHECKPOINT = MODELS_DIR / "classifier.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--samples-per-epoch", type=int, default=27_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-h", type=int, default=144)
    p.add_argument("--input-w", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real-crop-prob", type=float, default=0.5,
                   help="Probabilité d'utiliser un crop réel (si dispo) au lieu d'un template synthétique")
    p.add_argument("--no-bg-patches", action="store_true",
                   help="Désactiver les bg patches (non_card class)")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Checkpoint pour initialiser les poids (fine-tuning)")
    p.add_argument("--channels", type=int, nargs=4, default=[64, 128, 256, 512],
                   help="Largeurs des 4 stages (default teacher [64,128,256,512] ~11.2M)")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_ok, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        total_ok += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    return total_loss / total, total_ok / total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    if device.type == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")

    bg_dir = None if args.no_bg_patches else (BG_PATCHES_DIR if BG_PATCHES_DIR.exists() else None)
    real_dir = REAL_CROPS_DIR if REAL_CROPS_DIR.exists() else None

    train_ds = UnoTemplateDataset(
        TEMPLATES_DIR,
        bg_patches_dir=bg_dir,
        real_crops_dir=real_dir,
        real_crop_prob=args.real_crop_prob,
        n_samples=args.samples_per_epoch,
        input_size=(args.input_h, args.input_w),
        seed=args.seed,
    )
    val_ds = UnoTemplateDataset(
        TEMPLATES_DIR,
        bg_patches_dir=bg_dir,
        real_crops_dir=real_dir,
        real_crop_prob=args.real_crop_prob,
        n_samples=2_700,
        input_size=(args.input_h, args.input_w),
        seed=args.seed + 99_999,
    )
    print(f"Templates: {len(train_ds.templates)} cartes + "
          f"{train_ds.n_real_crops_total} real crops + "
          f"{len(train_ds.bg_patches)} bg patches")
    print(f"Train    : {len(train_ds)} samples/epoch | Val : {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = UnoCNN(num_classes=NUM_CLASSES, channels=tuple(args.channels)).to(device)
    n_params = count_parameters(model)
    print(f"Model    : UnoCNN(channels={args.channels}) | {n_params:,} params ({n_params / 1e6:.2f} M)")

    if args.init_from is not None and args.init_from.exists():
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        # Charge les poids partiellement si le nombre de classes diffère (ex: 55 → 54)
        state = ckpt["state_dict"]
        own_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
        model.load_state_dict(own_state)
        print(f"Init from {args.init_from.name} : {loaded}/{len(state)} tensors loaded")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, running_ok, running_n = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_ok += int((logits.argmax(1) == y).sum().item())
            running_n += y.size(0)

        train_loss = running_loss / running_n
        train_acc = running_ok / running_n
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        print(
            f"epoch {epoch:>3d}/{args.epochs} | "
            f"loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} | "
            f"{dt:.1f}s"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc, "time_s": dt}
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_size": (args.input_h, args.input_w),
                    "num_classes": NUM_CLASSES,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "history": history,
                },
                CHECKPOINT,
            )

    print(f"\n=> Meilleure val_acc : {best_val_acc:.3f}")
    print(f"   Checkpoint : {CHECKPOINT}")


if __name__ == "__main__":
    main()

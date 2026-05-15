"""Phase 4 V3 — Entraînement du corner-classifier (54 classes).

Option A (retenue) : from scratch, 50 epochs, LR 3e-4 cosine.

Option B (distillation depuis le classifier V2 11.2M) : NON applicable ici et
volontairement écartée. Raison empirique (Phase 1) : le teacher full-card
n'atteint que 5.9 % d'accuracy sur des crops de COINS (out-of-distribution).
Distiller depuis un teacher quasi-aléatoire sur l'input du student
injecterait du bruit au lieu de connaissance. La distillation suppose un
teacher compétent sur le MÊME input — ce n'est pas le cas. On documente ce
choix dans le rapport plutôt que d'implémenter une distillation invalide.

Validation : holdout des coins issus de real_crops (domaine réel = ce qui
compte à l'inférence).

Usage :
  python scripts/train_corner_classifier.py --epochs 50
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v3.corner_classifier import CornerClassifier, count_parameters  # noqa: E402
from src.v3.corner_classifier_dataset import CornerClassifierDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_corner_clf")

CKPT = ROOT / "checkpoints" / "v3_corner_classifier.pth"
LOG_DIR = ROOT / "logs"


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    tot_l = tot_ok = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logit = model(x)
        tot_l += crit(logit, y).item()
        tot_ok += int((logit.argmax(1) == y).sum())
        tot += y.numel()
    return tot_l / max(tot, 1), tot_ok / max(tot, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--samples-per-epoch", type=int, default=30000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--channels", type=int, nargs=4, default=[48, 96, 160, 224])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.parent.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    train_ds = CornerClassifierDataset(ROOT, augment=True,
                                       n_samples=args.samples_per_epoch,
                                       seed=args.seed)
    # Validation : real_crops uniquement (domaine réel), sans augment
    val_ds = CornerClassifierDataset(ROOT, augment=False,
                                     n_samples=3000, seed=args.seed + 5)
    val_ds.by_class = {k: [it for it in v if it[0] == "rc"]
                       for k, v in val_ds.by_class.items()}
    val_ds.by_class = {k: v for k, v in val_ds.by_class.items() if v}
    val_ds.classes = sorted(val_ds.by_class.keys())
    logger.info("Train classes=%d | Val (real_crops) classes=%d",
                len(train_ds.classes), len(val_ds.classes))

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True,
                    pin_memory=device.type == "cuda",
                    persistent_workers=args.num_workers > 0)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                    persistent_workers=args.num_workers > 0)

    model = CornerClassifier(channels=tuple(args.channels)).to(device)
    logger.info("CornerClassifier %.2fM params", count_parameters(model) / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.01)
    crit = nn.CrossEntropyLoss()
    best_acc = -1.0
    log_f = (LOG_DIR / "corner_clf.log").open("w")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        s_loss = s_ok = s_n = 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logit = model(x)
            loss = crit(logit, y)
            loss.backward()
            opt.step()
            s_loss += loss.item() * y.size(0)
            s_ok += int((logit.argmax(1) == y).sum())
            s_n += y.size(0)
        sched.step()
        vl_loss, vl_acc = evaluate(model, vl, device)
        dt = time.time() - t0
        msg = (f"ep {ep:3d}/{args.epochs} train_loss {s_loss/s_n:.4f} "
               f"acc {s_ok/s_n:.3f} | val_real loss {vl_loss:.4f} "
               f"acc {vl_acc:.3f} | {dt:.1f}s")
        logger.info(msg)
        log_f.write(msg + "\n"); log_f.flush()
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({"state_dict": model.state_dict(),
                        "channels": list(args.channels),
                        "epoch": ep, "val_acc": vl_acc}, CKPT)
    log_f.close()
    logger.info("Best val_real acc %.3f -> %s", best_acc, CKPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

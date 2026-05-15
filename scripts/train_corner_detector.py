"""Phase 3 V3 — Entraînement du détecteur de coins.

Phase A : synth + réel (real_ratio 0.30), 50 epochs, LR 1e-3 -> 1e-5 cosine
Phase B : fine-tune réel dominant (real_ratio 0.70), 15 epochs, LR 1e-4

Loss : focal (heatmap, style CenterNet) + L1 (offsets, pixels positifs).
Métriques : précision/rappel des pics détectés (tolérance 15 px).

Usage :
  python scripts/train_corner_detector.py --phase A --epochs 50
  python scripts/train_corner_detector.py --phase B --epochs 15 \
      --init checkpoints/v3_corner_detector.pth --lr 1e-4 --real-ratio 0.70
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v3.corner_dataset import (CornerDataset, DOWNSAMPLE,  # noqa: E402
                                   GRID_H, GRID_W)
from src.v3.corner_detector import CornerDetector, count_parameters  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_corner_det")

DATA = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               a: float = 2.0, b: float = 4.0) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    pos = (target == 1.0).float()
    neg = (target < 1.0).float()
    eps = 1e-6
    pl = -((1 - pred) ** a) * torch.log(pred + eps) * pos
    nl = -((1 - target) ** b) * (pred ** a) * torch.log(1 - pred + eps) * neg
    n = pos.sum().clamp(min=1)
    return (pl.sum() + nl.sum()) / n


def off_loss(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d = (pred - tgt).abs() * mask
    return d.sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def corner_pr(model, loader, device, tol_px: int = 15, thr: float = 0.3
              ) -> tuple[float, float]:
    """Précision/rappel des pics (tolérance tol_px en pixels image)."""
    model.eval()
    tp = fp = fn = 0
    tol_grid = tol_px / DOWNSAMPLE
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        hm = torch.sigmoid(out["heatmap"])
        pooled = F.max_pool2d(hm, 3, 1, 1)
        peaks = (hm == pooled) & (hm > thr)
        for i in range(img.size(0)):
            pk = peaks[i, 0].nonzero(as_tuple=False).cpu().numpy()  # (N,2) y,x
            gt = batch["mask"][i, 0].nonzero(as_tuple=False).numpy()
            used = set()
            for (gy, gx) in gt:
                hit = False
                for j, (py, px) in enumerate(pk):
                    if j in used:
                        continue
                    if abs(py - gy) <= tol_grid and abs(px - gx) <= tol_grid:
                        used.add(j); hit = True; break
                if hit:
                    tp += 1
                else:
                    fn += 1
            fp += len(pk) - len(used)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["A", "B"], default="A")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--samples-per-epoch", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--real-ratio", type=float, default=0.30)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--channels", type=int, nargs=4, default=[40, 80, 160, 320])
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    logger.info("Device=%s phase=%s real_ratio=%.2f", device, args.phase, args.real_ratio)

    train_ds = CornerDataset(
        DATA / "synthetic" / "images", DATA / "synthetic" / "labels",
        DATA / "train_images", DATA / "yolo_annotations",
        real_ratio=args.real_ratio, augment=True,
        n_samples=args.samples_per_epoch, seed=args.seed)
    val_ds = CornerDataset(
        DATA / "synthetic" / "images", DATA / "synthetic" / "labels",
        DATA / "train_images", DATA / "yolo_annotations",
        real_ratio=1.0, augment=False, n_samples=160, seed=args.seed + 7)

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True,
                    pin_memory=device.type == "cuda",
                    persistent_workers=args.num_workers > 0)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                    persistent_workers=args.num_workers > 0)

    model = CornerDetector(channels=tuple(args.channels)).to(device)
    logger.info("CornerDetector %.2fM params", count_parameters(model) / 1e6)
    if args.init and args.init.exists():
        model.load_state_dict(torch.load(args.init, map_location=device)["state_dict"])
        logger.info("Init from %s", args.init.name)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.01)
    ckpt_path = CKPT_DIR / "v3_corner_detector.pth"
    best_f1 = -1.0  # critère = F1(P,R), pas recall seul (saturait à 0.999)
    log_f = (LOG_DIR / f"corner_det_phase{args.phase}.log").open("w")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        s_loss = s_hm = s_off = 0.0
        nb = 0
        for batch in tl:
            img = batch["image"].to(device, non_blocking=True)
            hm_t = batch["heatmap"].to(device, non_blocking=True)
            off_t = batch["offset"].to(device, non_blocking=True)
            msk = batch["mask"].to(device, non_blocking=True)
            out = model(img)
            l_hm = focal_loss(out["heatmap"], hm_t)
            l_off = off_loss(out["offset"], off_t, msk)
            loss = l_hm + l_off
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            s_loss += loss.item(); s_hm += l_hm.item(); s_off += l_off.item()
            nb += 1
        sched.step()
        prec, rec = corner_pr(model, vl, device)
        dt = time.time() - t0
        msg = (f"ep {ep:3d}/{args.epochs} loss {s_loss/nb:.4f} "
               f"(hm {s_hm/nb:.4f} off {s_off/nb:.4f}) | "
               f"val P={prec:.3f} R={rec:.3f} | {dt:.1f}s")
        logger.info(msg)
        log_f.write(msg + "\n"); log_f.flush()
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"state_dict": model.state_dict(),
                        "channels": list(args.channels),
                        "epoch": ep, "prec": prec, "rec": rec, "f1": f1},
                       ckpt_path)
    log_f.close()
    logger.info("Best F1 %.3f -> %s", best_f1, ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

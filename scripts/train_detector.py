"""Entraîne le détecteur axis-aligned, channels configurables.

Stratégie 2-phase :
- pretrain : mix synth (5000) + real (81) avec ratio 80/20, channels [40,80,160,320]
- finetune : real seul, lr plus bas

Loss : focal sur obj + L1 sur bbox aux pixels centraux.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import CardDetector, count_parameters  # noqa: E402
from src.detector_dataset import CardDetectionDataset  # noqa: E402

DATA_DIR = ROOT / "data"
REAL_IMG_DIR = DATA_DIR / "train_images"
REAL_ANN_DIR = DATA_DIR / "yolo_annotations"
SYNTH_IMG_DIR = DATA_DIR / "synth_images"
SYNTH_ANN_DIR = DATA_DIR / "synth_annotations"
MODELS_DIR = ROOT / "outputs" / "models"


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    pos_mask = (target == 1.0).float()
    neg_mask = (target < 1.0).float()
    eps = 1e-6
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred + eps) * pos_mask
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred + eps) * neg_mask
    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def bbox_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    n_pos = mask.sum().clamp(min=1)
    return diff.sum() / n_pos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["pretrain", "finetune"], default="pretrain")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bbox-weight", type=float, default=5.0,
                   help="Poids relatif de la loss bbox vs objectness")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--channels", type=int, nargs=4, default=[40, 80, 160, 320],
                   help="Largeurs des 4 stages")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Checkpoint à charger (pour fine-tune)")
    p.add_argument("--out-name", type=str, default=None,
                   help="Nom du checkpoint (default: detector_<phase>.pt)")
    p.add_argument("--synth-weight", type=float, default=0.80,
                   help="Poids du synthétique dans le mix (0.80 = 80/20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Phase  : {args.phase}")

    # Datasets selon la phase
    if args.phase == "pretrain":
        img_dirs = [SYNTH_IMG_DIR, REAL_IMG_DIR]
        ann_dirs = [SYNTH_ANN_DIR, REAL_ANN_DIR]
        weights = [args.synth_weight, 1.0 - args.synth_weight]
    else:
        img_dirs = [REAL_IMG_DIR]
        ann_dirs = [REAL_ANN_DIR]
        weights = [1.0]

    train_ds = CardDetectionDataset(
        img_dirs, ann_dirs, weights=weights,
        augment=True, n_samples=args.samples_per_epoch, seed=args.seed,
    )
    val_ds = CardDetectionDataset(
        img_dirs, ann_dirs, weights=weights,
        augment=False, n_samples=200, seed=args.seed + 1,
    )
    print(f"Sources : {[str(d.name) for d in img_dirs]} (weights={weights})")
    print(f"Train samples/epoch : {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=args.num_workers > 0)

    model = CardDetector(channels=tuple(args.channels)).to(device)
    n_params = count_parameters(model)
    print(f"Detector channels={args.channels} : {n_params:,} params ({n_params / 1e6:.2f} M)")

    if args.init_from is not None and args.init_from.exists():
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Init from : {args.init_from.name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"detector_{args.phase}.pt"
    out_path = MODELS_DIR / out_name
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_obj_sum = 0.0
        train_bbox_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            obj_t = batch["obj_target"].to(device, non_blocking=True)
            bbox_t = batch["bbox_target"].to(device, non_blocking=True)
            mask = batch["bbox_mask"].to(device, non_blocking=True)

            out = model(img)
            loss_obj = focal_loss(out["obj"], obj_t)
            loss_bbox = bbox_loss(out["bbox"], bbox_t, mask)
            loss = loss_obj + args.bbox_weight * loss_bbox

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_obj_sum += loss_obj.item()
            train_bbox_sum += loss_bbox.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        train_obj = train_obj_sum / max(n_batches, 1)
        train_bbox = train_bbox_sum / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device, non_blocking=True)
                obj_t = batch["obj_target"].to(device, non_blocking=True)
                bbox_t = batch["bbox_target"].to(device, non_blocking=True)
                mask = batch["bbox_mask"].to(device, non_blocking=True)
                out = model(img)
                lo = focal_loss(out["obj"], obj_t)
                lb = bbox_loss(out["bbox"], bbox_t, mask)
                val_loss_sum += (lo + args.bbox_weight * lb).item()
                val_n += 1
        val_loss = val_loss_sum / max(val_n, 1)
        scheduler.step()
        dt = time.time() - t0

        print(f"epoch {epoch:>3d}/{args.epochs} | "
              f"train {train_loss:.4f} (obj {train_obj:.4f} bb {train_bbox:.4f}) | "
              f"val {val_loss:.4f} | {dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": val_loss,
                "channels": list(args.channels),
                "phase": args.phase,
            }, out_path)

    print(f"\n=> Best val_loss : {best_val_loss:.4f}")
    print(f"   Checkpoint : {out_path}")


if __name__ == "__main__":
    main()

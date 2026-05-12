"""Entraîne le détecteur de cartes anchor-free sur les 81 images annotées.

Loss :
- Focal loss sur la heatmap d'objectness
- L1 loss sur la bbox (cx, cy, w, h) — uniquement aux pixels positifs (mask)

Sortie : outputs/models/detector.pt
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
TRAIN_IMG_DIR = DATA_DIR / "train_images"
ANNOT_DIR = DATA_DIR / "yolo_annotations"
CHECKPOINT = ROOT / "outputs" / "models" / "detector.pt"


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Focal loss style CenterNet sur la heatmap.

    target ∈ [0, 1] (heatmap gaussienne). Le pixel central a target=1.0,
    les pixels voisins ont une valeur > 0 décroissante.
    """
    pred = torch.sigmoid(logits)
    pos_mask = (target == 1.0).float()
    neg_mask = (target < 1.0).float()

    eps = 1e-6
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred + eps) * pos_mask
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred + eps) * neg_mask

    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def bbox_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss sur (cx, cy, w, h), pondérée par le mask des positifs."""
    diff = (pred - target).abs() * mask
    n_pos = mask.sum().clamp(min=1)
    return diff.sum() / n_pos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--samples-per-epoch", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bbox-weight", type=float, default=5.0,
                   help="Poids relatif de la loss bbox vs objectness")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_ds = CardDetectionDataset(
        TRAIN_IMG_DIR, ANNOT_DIR,
        augment=True, n_samples=args.samples_per_epoch, seed=args.seed,
    )
    val_ds = CardDetectionDataset(
        TRAIN_IMG_DIR, ANNOT_DIR,
        augment=False, n_samples=81, seed=args.seed + 1,
    )
    print(f"Train images : {len(train_ds.image_paths)}")
    print(f"Train samples/epoch : {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=args.num_workers > 0)

    model = CardDetector().to(device)
    n_params = count_parameters(model)
    print(f"Detector : {n_params:,} params ({n_params / 1e6:.2f} M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
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

        train_loss = train_loss_sum / n_batches
        train_obj = train_obj_sum / n_batches
        train_bbox = train_bbox_sum / n_batches

        # Validation
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
              f"train_loss {train_loss:.4f} (obj {train_obj:.4f} bbox {train_bbox:.4f}) | "
              f"val_loss {val_loss:.4f} | {dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, CHECKPOINT)

    print(f"\n=> Best val_loss : {best_val_loss:.4f}")
    print(f"   Checkpoint : {CHECKPOINT}")


if __name__ == "__main__":
    main()

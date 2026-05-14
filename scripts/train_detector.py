"""Entraîne le détecteur OBB multi-niveau (CenterNet++ avec FPN P3/P4/P5).

Stratégie d'entraînement en 2 phases :
  Phase 1 : pre-train sur dataset synthétique (5k images avec OBB exactes)
  Phase 2 : fine-tune sur 81 images réelles + synthétique (mix 50/50)

Loss : (focal_obj + bbox_w * L1_bbox + angle_w * smooth_L1_angle) sur 3 niveaux

Usage :
    # Phase 1 : pre-train sur synth uniquement
    python scripts/train_detector.py --phase pretrain --epochs 40

    # Phase 2 : fine-tune avec mix réel + synth
    python scripts/train_detector.py --phase finetune --epochs 40 \
        --init-from outputs/models/detector_pretrain.pt
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
from src.detector_dataset import LEVELS, OBBDetectionDataset  # noqa: E402

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


def bbox_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    n_pos = mask.sum().clamp(min=1)
    return diff.sum() / n_pos


def angle_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Smooth L1 sur (sin θ, cos θ) au pixel central."""
    diff = F.smooth_l1_loss(pred, target, reduction="none") * mask
    n_pos = mask.sum().clamp(min=1)
    return diff.sum() / n_pos


def compute_loss(out: dict, batch: dict, bbox_w: float = 5.0,
                 angle_w: float = 1.0) -> tuple[torch.Tensor, dict]:
    total = torch.tensor(0.0, device=next(iter(out.values())).device)
    breakdown: dict[str, float] = {}
    for lvl in LEVELS:
        l_obj = focal_loss(out[f"obj_{lvl}"], batch[f"obj_{lvl}"])
        l_bb = bbox_l1_loss(out[f"bbox_{lvl}"], batch[f"bbox_{lvl}"], batch[f"mask_{lvl}"])
        l_ang = angle_loss(out[f"angle_{lvl}"], batch[f"angle_{lvl}"], batch[f"mask_{lvl}"])
        total = total + l_obj + bbox_w * l_bb + angle_w * l_ang
        breakdown[f"{lvl}_obj"] = l_obj.item()
        breakdown[f"{lvl}_bb"] = l_bb.item()
        breakdown[f"{lvl}_ang"] = l_ang.item()
    return total, breakdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bbox-weight", type=float, default=5.0)
    p.add_argument("--angle-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-from", type=Path, default=None,
                   help="checkpoint à charger pour le fine-tune")
    p.add_argument("--out-name", type=str, default=None,
                   help="nom du checkpoint (default: detector_<phase>.pt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Phase  : {args.phase}")

    if args.phase == "pretrain":
        # 100% synthétique
        img_dirs = [SYNTH_IMG_DIR]
        ann_dirs = [SYNTH_ANN_DIR]
        weights = [1.0]
    else:
        # 50% synth + 50% réel
        img_dirs = [SYNTH_IMG_DIR, REAL_IMG_DIR]
        ann_dirs = [SYNTH_ANN_DIR, REAL_ANN_DIR]
        weights = [0.5, 0.5]

    train_ds = OBBDetectionDataset(
        img_dirs, ann_dirs, weights=weights,
        augment=True, n_samples=args.samples_per_epoch, seed=args.seed,
    )
    val_ds = OBBDetectionDataset(
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

    model = CardDetector().to(device)
    n_params = count_parameters(model)
    print(f"Detector : {n_params:,} params ({n_params / 1e6:.2f} M)")

    if args.init_from is not None and args.init_from.exists():
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Init from : {args.init_from}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"detector_{args.phase}.pt"
    out_path = MODELS_DIR / out_name
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_breakdown_sum: dict[str, float] = {}
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(batch["image"])
            loss, breakdown = compute_loss(out, batch,
                                            bbox_w=args.bbox_weight,
                                            angle_w=args.angle_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            for k, v in breakdown.items():
                train_breakdown_sum[k] = train_breakdown_sum.get(k, 0.0) + v
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        train_breakdown = {k: v / max(n_batches, 1) for k, v in train_breakdown_sum.items()}

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                out = model(batch["image"])
                loss, _ = compute_loss(out, batch,
                                        bbox_w=args.bbox_weight,
                                        angle_w=args.angle_weight)
                val_loss_sum += loss.item()
                val_n += 1
        val_loss = val_loss_sum / max(val_n, 1)
        scheduler.step()
        dt = time.time() - t0

        # Brief log
        obj_sum = sum(train_breakdown[f"{lvl}_obj"] for lvl in LEVELS)
        bb_sum = sum(train_breakdown[f"{lvl}_bb"] for lvl in LEVELS)
        ang_sum = sum(train_breakdown[f"{lvl}_ang"] for lvl in LEVELS)
        print(f"epoch {epoch:>3d}/{args.epochs} | "
              f"train {train_loss:.4f} (obj {obj_sum:.3f} bb {bb_sum:.3f} ang {ang_sum:.3f}) | "
              f"val {val_loss:.4f} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": val_loss,
                "phase": args.phase,
            }, out_path)

    print(f"\n=> Best val_loss : {best_val:.4f}")
    print(f"   Checkpoint : {out_path}")


if __name__ == "__main__":
    main()

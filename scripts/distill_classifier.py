"""Distillation knowledge : teacher 11.2M (classifier.pt) -> student 6.3M.

Loss combinée :
    L = alpha * CE(student_logits, hard_labels)
      + (1 - alpha) * KL(softmax(student/T) || softmax(teacher/T)) * T^2

T (temperature) lisse les distributions du teacher pour transférer plus
d'information sur les classes secondaires que juste le top-1.

Usage :
    python scripts/distill_classifier.py --epochs 30 --alpha 0.3 --temperature 4.0

Sortie : outputs/models/classifier_student.pt (le student 6.3M).
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

from src.config import MODELS_DIR, NUM_CLASSES  # noqa: E402
from src.dataset import UnoTemplateDataset  # noqa: E402
from src.model import UnoCNN, count_parameters  # noqa: E402

TEMPLATES_DIR = ROOT / "data" / "card_templates"
BG_PATCHES_DIR = ROOT / "data" / "bg_patches"
REAL_CROPS_DIR = ROOT / "data" / "real_crops"
TEACHER_CHECKPOINT = MODELS_DIR / "classifier.pt"
STUDENT_CHECKPOINT = MODELS_DIR / "classifier_student.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--samples-per-epoch", type=int, default=27_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-h", type=int, default=144)
    p.add_argument("--input-w", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real-crop-prob", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.3,
                   help="poids du hard CE vs distillation KL (0=full distill, 1=full hard)")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="température de lissage du teacher")
    p.add_argument("--student-channels", type=int, nargs=4,
                   default=[48, 96, 192, 384],
                   help="channels du student (default ~6M)")
    p.add_argument("--no-bg-patches", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device
             ) -> tuple[float, float]:
    model.eval()
    total_loss, total_ok, total = 0.0, 0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += crit(logits, y).item()
        total_ok += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    return total_loss / total, total_ok / total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Datasets
    bg_dir = None if args.no_bg_patches else (BG_PATCHES_DIR if BG_PATCHES_DIR.exists() else None)
    real_dir = REAL_CROPS_DIR if REAL_CROPS_DIR.exists() else None
    train_ds = UnoTemplateDataset(
        TEMPLATES_DIR, bg_patches_dir=bg_dir, real_crops_dir=real_dir,
        real_crop_prob=args.real_crop_prob,
        n_samples=args.samples_per_epoch,
        input_size=(args.input_h, args.input_w), seed=args.seed,
    )
    val_ds = UnoTemplateDataset(
        TEMPLATES_DIR, bg_patches_dir=bg_dir, real_crops_dir=real_dir,
        real_crop_prob=args.real_crop_prob,
        n_samples=2_700,
        input_size=(args.input_h, args.input_w), seed=args.seed + 99_999,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=args.num_workers > 0)

    # Teacher (frozen)
    if not TEACHER_CHECKPOINT.exists():
        raise FileNotFoundError(f"Teacher missing : {TEACHER_CHECKPOINT}")
    ckpt_t = torch.load(TEACHER_CHECKPOINT, map_location=device, weights_only=False)
    # Detect teacher channels from state_dict (look at the stem weight)
    stem_weight = ckpt_t["state_dict"]["stem.0.weight"]
    teacher_c1 = stem_weight.shape[0]
    teacher_channels = (teacher_c1, teacher_c1 * 2, teacher_c1 * 4, teacher_c1 * 8)
    teacher = UnoCNN(num_classes=ckpt_t.get("num_classes", NUM_CLASSES),
                     channels=teacher_channels)
    teacher.load_state_dict(ckpt_t["state_dict"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_t = count_parameters(teacher)
    print(f"Teacher  : {n_t / 1e6:.2f}M params (channels={teacher_channels})")

    # Student
    student = UnoCNN(num_classes=NUM_CLASSES,
                     channels=tuple(args.student_channels)).to(device)
    n_s = count_parameters(student)
    print(f"Student  : {n_s / 1e6:.2f}M params (channels={args.student_channels})")
    print(f"Distill  : alpha={args.alpha} T={args.temperature}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history: list[dict] = []

    T = args.temperature
    for epoch in range(1, args.epochs + 1):
        student.train()
        t0 = time.time()
        running_loss, running_ok, running_n = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(x)
            soft_targets = F.softmax(teacher_logits / T, dim=1)

            # Student forward
            student_logits = student(x)
            student_log_soft = F.log_softmax(student_logits / T, dim=1)

            # Combined loss
            l_hard = ce_loss(student_logits, y)
            l_soft = F.kl_div(student_log_soft, soft_targets, reduction="batchmean") * (T * T)
            loss = args.alpha * l_hard + (1 - args.alpha) * l_soft

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_ok += int((student_logits.argmax(1) == y).sum().item())
            running_n += y.size(0)

        train_loss = running_loss / running_n
        train_acc = running_ok / running_n
        val_loss, val_acc = evaluate(student, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        print(f"epoch {epoch:>3d}/{args.epochs} | "
              f"loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} | "
              f"{dt:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "time_s": dt})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": student.state_dict(),
                "input_size": (args.input_h, args.input_w),
                "num_classes": NUM_CLASSES,
                "channels": list(args.student_channels),
                "epoch": epoch, "val_acc": val_acc, "history": history,
                "distilled_from": str(TEACHER_CHECKPOINT.name),
            }, STUDENT_CHECKPOINT)

    print(f"\n=> Meilleure val_acc student : {best_val_acc:.3f}")
    print(f"   Checkpoint : {STUDENT_CHECKPOINT}")


if __name__ == "__main__":
    main()

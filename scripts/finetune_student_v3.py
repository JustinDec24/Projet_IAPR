"""Fine-tune le student classifier V2 (6.31M) sur les real_crops_v3 enrichis
(406 crops, 54 classes, mieux localisés via corner-detector R=0.99).

Init depuis classifier_student.pt. Sortie : classifier_student_v3.pt
(ne touche pas aux modèles V2 de production).
"""
from __future__ import annotations

import argparse
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("finetune_v3")

TPL = ROOT / "data" / "card_templates"
INIT = MODELS_DIR / "classifier_student.pt"


@torch.no_grad()
def evaluate(m, loader, dev):
    m.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    tl = ok = tot = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        o = m(x)
        tl += crit(o, y).item()
        ok += int((o.argmax(1) == y).sum())
        tot += y.numel()
    return tl / tot, ok / tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--samples-per-epoch", type=int, default=27000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--real-crop-prob", type=float, default=0.65)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crops-dir", type=str, default="data/real_crops_v3")
    ap.add_argument("--out", type=str, default="classifier_student_v3.pt")
    args = ap.parse_args()
    RC = ROOT / args.crops_dir
    OUT = MODELS_DIR / args.out
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(INIT, map_location=dev, weights_only=False)
    ch = tuple(ck.get("channels", (48, 96, 192, 384)))
    in_sz = tuple(ck["input_size"])
    logger.info("Student init channels=%s input=%s", ch, in_sz)

    tr = UnoTemplateDataset(TPL, bg_patches_dir=None, real_crops_dir=RC,
                            real_crop_prob=args.real_crop_prob,
                            n_samples=args.samples_per_epoch,
                            input_size=in_sz, seed=args.seed)
    va = UnoTemplateDataset(TPL, bg_patches_dir=None, real_crops_dir=RC,
                            real_crop_prob=args.real_crop_prob,
                            n_samples=2700, input_size=in_sz,
                            seed=args.seed + 999)
    logger.info("real_crops_v3 : %d crops", tr.n_real_crops_total)
    tld = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers, drop_last=True,
                     pin_memory=dev.type == "cuda",
                     persistent_workers=args.num_workers > 0)
    vld = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers,
                     pin_memory=dev.type == "cuda",
                     persistent_workers=args.num_workers > 0)

    model = UnoCNN(num_classes=NUM_CLASSES, channels=ch).to(dev)
    model.load_state_dict(ck["state_dict"])
    logger.info("Loaded student %.2fM params", count_parameters(model) / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                     eta_min=args.lr * 0.02)
    crit = nn.CrossEntropyLoss()
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sl = sok = sn = 0
        for x, y in tld:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            o = model(x)
            loss = crit(o, y)
            loss.backward()
            opt.step()
            sl += loss.item() * y.size(0)
            sok += int((o.argmax(1) == y).sum())
            sn += y.size(0)
        sch.step()
        vl, va_acc = evaluate(model, vld, dev)
        logger.info("ep %2d/%d train_loss %.4f acc %.3f | val_loss %.4f acc %.3f | %.1fs",
                    ep, args.epochs, sl / sn, sok / sn, vl, va_acc, time.time() - t0)
        if va_acc > best:
            best = va_acc
            torch.save({"state_dict": model.state_dict(),
                        "input_size": list(in_sz), "num_classes": NUM_CLASSES,
                        "channels": list(ch), "epoch": ep, "val_acc": va_acc},
                       OUT)
    logger.info("Best val_acc %.3f -> %s", best, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

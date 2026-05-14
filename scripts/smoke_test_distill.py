"""Smoke test : 1 forward + 1 backward de distillation pour traquer le crash."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, NUM_CLASSES
from src.dataset import UnoTemplateDataset
from src.model import UnoCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets (mêmes args que distill_classifier)
ds = UnoTemplateDataset(
    ROOT / "data" / "card_templates",
    bg_patches_dir=ROOT / "data" / "bg_patches",
    real_crops_dir=ROOT / "data" / "real_crops",
    real_crop_prob=0.5,
    n_samples=10,
    input_size=(144, 96),
    seed=0,
)
print(f"Dataset ok : {ds.n_real_crops_total} real crops, {len(ds.bg_patches)} bg patches")

# Test getting a few items
for i in range(5):
    try:
        x, y = ds[i]
        print(f"item {i}: x.shape={x.shape} y={y}")
    except Exception as e:
        print(f"item {i} FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Teacher
ckpt = torch.load(MODELS_DIR / "classifier.pt", map_location=device, weights_only=False)
stem = ckpt["state_dict"]["stem.0.weight"]
c1 = stem.shape[0]
channels_t = (c1, c1 * 2, c1 * 4, c1 * 8)
teacher = UnoCNN(num_classes=ckpt.get("num_classes", NUM_CLASSES), channels=channels_t)
teacher.load_state_dict(ckpt["state_dict"])
teacher.to(device).eval()
print(f"Teacher loaded : channels={channels_t}, num_classes={ckpt.get('num_classes')}")

# Student
student = UnoCNN(num_classes=NUM_CLASSES, channels=(48, 96, 192, 384)).to(device)
print(f"Student : channels=(48,96,192,384)")

# Mini batch
xs, ys = [], []
for i in range(8):
    x, y = ds[i]
    xs.append(x)
    ys.append(y)
batch_x = torch.stack(xs).to(device)
batch_y = torch.tensor(ys).to(device)
print(f"Batch : {batch_x.shape}")

# Forward
with torch.no_grad():
    teacher_logits = teacher(batch_x)
student_logits = student(batch_x)
print(f"Teacher logits : {teacher_logits.shape}")
print(f"Student logits : {student_logits.shape}")

# Distill loss
T = 4.0
soft_t = F.softmax(teacher_logits / T, dim=1)
log_soft_s = F.log_softmax(student_logits / T, dim=1)
l_hard = nn.CrossEntropyLoss()(student_logits, batch_y)
l_soft = F.kl_div(log_soft_s, soft_t, reduction="batchmean") * (T * T)
loss = 0.3 * l_hard + 0.7 * l_soft
print(f"Loss : {loss.item():.4f}  (hard={l_hard.item():.4f}, soft={l_soft.item():.4f})")
loss.backward()
print("Backward OK")

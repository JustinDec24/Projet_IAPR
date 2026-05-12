"""Constantes partagées : classes UNO, géométrie des joueurs, chemins."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train_images"
TEST_DIR = DATA_DIR / "test_images"
REFERENCE_DIR = DATA_DIR / "reference_images"
TRAIN_CSV = DATA_DIR / "train.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

COLORS = ("r", "g", "b", "y")
NUMBERS = tuple(str(n) for n in range(10))
COLOR_ACTIONS = ("skip", "reverse", "draw_2")
WILD_CARDS = ("wild", "draw_4")

CARD_CLASSES = tuple(
    [f"{c}_{n}" for c in COLORS for n in NUMBERS]
    + [f"{c}_{a}" for c in COLORS for a in COLOR_ACTIONS]
    + list(WILD_CARDS)
)
NUM_CLASSES = len(CARD_CLASSES)  # 54
CLASS_TO_IDX = {name: i for i, name in enumerate(CARD_CLASSES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

# Kept for backward compat (inference checks them — they're no-ops with 54 classes).
NON_CARD_LABEL = "non_card"
NON_CARD_IDX = -1

PLAYERS = ("p1", "p2", "p3", "p4")
PLAYER_SIDES = {
    "p1": "bottom",
    "p2": "right",
    "p3": "top",
    "p4": "left",
}

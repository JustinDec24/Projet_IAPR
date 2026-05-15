"""Phase 5 V3 — Définition des 5 zones (4 joueurs + centre).

Coordonnées relatives [0,1] de l'image, chargées depuis configs/v3_synthetic.yaml
(mêmes zones que la génération synthétique pour cohérence train/inférence).
0 paramètre.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ZONES = {
    "p1": (0.25, 0.70, 0.75, 1.00),   # bottom
    "p2": (0.70, 0.20, 1.00, 0.80),   # right
    "p3": (0.25, 0.00, 0.75, 0.30),   # top
    "p4": (0.00, 0.20, 0.30, 0.80),   # left
    "center": (0.35, 0.35, 0.65, 0.65),
}


def load_zones(config_path: str | Path | None = None) -> dict[str, tuple]:
    """Renvoie {zone: (x0,y0,x1,y1)} relatifs [0,1]."""
    if config_path is None:
        config_path = ROOT / "configs" / "v3_synthetic.yaml"
    p = Path(config_path)
    if p.exists():
        cfg = yaml.safe_load(p.read_text())
        z = cfg.get("scene", {}).get("zones")
        if z:
            return {k: tuple(v) for k, v in z.items()}
    return dict(_DEFAULT_ZONES)


def which_zone(cx: float, cy: float, img_w: int, img_h: int,
                zones: dict[str, tuple] | None = None) -> str:
    """Assigne un point (pixels) à une zone.

    Priorité au centre (zone la plus petite). Si hors de toute zone joueur,
    on rattache à la zone joueur la plus proche du point (robustesse bords).
    """
    if zones is None:
        zones = load_zones()
    fx, fy = cx / max(img_w, 1), cy / max(img_h, 1)

    cz = zones["center"]
    if cz[0] <= fx <= cz[2] and cz[1] <= fy <= cz[3]:
        return "center"
    for name in ("p1", "p2", "p3", "p4"):
        x0, y0, x1, y1 = zones[name]
        if x0 <= fx <= x1 and y0 <= fy <= y1:
            return name
    # Fallback : zone joueur dont le centre est le plus proche
    best, best_d = "p1", 1e9
    for name in ("p1", "p2", "p3", "p4"):
        x0, y0, x1, y1 = zones[name]
        zcx, zcy = (x0 + x1) / 2, (y0 + y1) / 2
        d = (fx - zcx) ** 2 + (fy - zcy) ** 2
        if d < best_d:
            best_d, best = d, name
    return best

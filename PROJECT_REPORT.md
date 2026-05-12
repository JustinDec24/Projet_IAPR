# UNO Vision — Rapport de projet IAPR 2026

Compétition Kaggle [iapr-26-uno-vision-challenge](https://www.kaggle.com/competitions/iapr-26-uno-vision-challenge).

Pour chaque image de partie de UNO (4000×2662 px), prédire :
- la **carte centrale** posée au milieu ;
- le **joueur actif** (p1 / p2 / p3 / p4) — celui qui possède le jeton ;
- les **cartes en main** de chacun des 4 joueurs.

**Métrique** : `Score = 0.1·CenterAcc + 0.1·ActiveAcc + 0.8·F1` (F1 multi-ensemble sur les cartes des joueurs).

**Score final (train, 81 images)** : **0.764** (CenterAcc 0.901, ActiveAcc 0.716, F1 0.753).

---

## 1. Contraintes du projet

- Aucun dataset externe ni modèle pré-entraîné.
- **≤ 12M paramètres** au total sur tous les modèles du pipeline (confirmé sur le forum Moodle : la limite s'applique à la somme, pas à chaque modèle individuellement).
- 81 images d'entraînement annotées (CSV `train.csv`) + 159 images de test.

Budget actuel :

| Modèle | Paramètres |
|---|---|
| Détecteur (CenterNet-like, channels [16, 32, 64, 96]) | **0.53 M** |
| Classifieur (UnoCNN, ResNet-18-light) | **11.20 M** |
| **Total** | **11.73 M** ✓ |

---

## 2. Pipeline hybride

```
image
  │
  ├── détection token (jaune ou noir, threshold V adaptatif)
  │
  ├── détection cartes (hybride) ─┬── heuristique HSV → cartes en zone centrale
  │                                └── détecteur appris → cartes en zone joueurs
  │
  ├── classification CNN (54 classes, TTA 8×)
  │
  └── assignation par géométrie (angle depuis le centre image)
       → carte centrale, cartes p1/p2/p3/p4, joueur actif
```

### 2.1. Templates (`src/templates.py`)

Les 54 cartes UNO sont extraites des `data/reference_images/` :
- 6 photos de cartes étalées (`L1000700`…`L1000766`) ;
- segmentation HSV + clustering par couleur dominante + per-color split ;
- 3 fallbacks hardcodés pour `L1000766` (cartes `y_5`, `y_4`, `r_4` impossibles à isoler par contour à cause d'un L-shape merge).

Sortie : 54 templates 200×300 dans `data/card_templates/`.

### 2.2. Détection heuristique (`src/detection.py`)

Deux stratégies fusionnées :

- **Saturation blur** : flou gaussien sur le canal S de HSV → seuillage → contours → rotated rect.
- **White ovals** : pour les fonds bruités, on cherche les ovales blancs (corner d'une carte UNO) enclos dans une zone fermée.
- **Per-color split** : si un blob a une aire > 1.2× la médiane ET ≥ 2 couleurs distinctes, on le split par k-means couleur (pour séparer les piles de cartes empilées).

Token actif :
- threshold adaptatif `V_median - 80` (fond clair → seuil ~150 capte le token noir ; fond bruité → seuil ~20 + fallback jaune).
- assignation au joueur le plus proche.

### 2.3. Détecteur appris (`src/detector.py`)

Architecture anchor-free style CenterNet :
- backbone : stem 3→16 conv stride 2, puis 3 stages (BasicBlocks 16→32, 32→64, 64→96), downsample total 16× ;
- head partagé 96→64, puis 2 conv 1×1 :
  - `obj` : heatmap d'objectness (logits, sigmoid au sample) ;
  - `bbox` : (cx, cy, w, h) normalisés [0, 1] de l'image entière, sigmoid.
- bias init du head obj à `-4.6` (prior `sigmoid(-4.6) ≈ 0.01`).

Entrée 512×352 → grille 32×22. **0.53 M params**.

#### Entraînement (`scripts/train_detector.py`)

- **Données** : 626 bboxes annotées manuellement dans LabelImg (1 classe « card »), format YOLO, sur les 81 images train.
- **Targets** :
  - `obj_target` : heatmap gaussienne 2D centrée sur chaque bbox, `np.maximum`-mergée si chevauchement ;
  - **fix critique** : `obj_target[cy, cx] = 1.0` explicitement au pixel central (sinon le max de la gaussienne discrète est < 1.0, et `pos_mask = (target == 1.0)` ne sélectionne aucun pixel → le modèle apprend uniquement à dire « tout zéro ») ;
  - `bbox_target` : (cx, cy, w, h) normalisés posés au pixel central ;
  - `bbox_mask` : 1 au pixel central, 0 ailleurs.
- **Loss** : focal loss CenterNet (α=2, β=4) sur `obj` + L1 sur `bbox` masquée, ratio 1 : 5.
- **Augmentations** : random scale+crop (0.65–1.0), flip horizontal, color jitter HSV, blur gaussien (p=0.3), bruit gaussien (p=0.4).
- **Hyperparams** : 80 epochs, batch 16, 1000 samples/epoch, AdamW lr=2e-3, cosine schedule.

#### Inférence (`src/detector_inference.py`)

- forward → `obj` heatmap + `bbox` regression ;
- seuillage `obj > 0.35` ;
- NMS custom (IoU 0.45) ;
- max 30 détections par image.

### 2.4. Classifieur (`src/model.py`, `src/dataset.py`, `scripts/train_classifier.py`)

**UnoCNN** = ResNet-18-light :
- stem (3→64, stride 2) ;
- 4 stages (64, 128, 256, 512 channels, 2 BasicBlocks chacun) ;
- GAP + Linear(512, 54).
- **11.20 M params**.

Entrée : crop 144×96 portrait (downscalé de 200×300).

**Données d'entraînement** :
- 54 templates synthétiques (cf. 2.1) × augmentations agressives :
  - rotation discrète {0°, 90°, 180°, 270°} (cartes UNO symétriques 180°) ;
  - perspective warp, scale, translation ;
  - color jitter HSV, brightness/contrast ;
  - bruit gaussien, JPEG compression simulée.
- **389 crops réels auto-labellisés** (cf. 2.5) — clé du gain principal.

**Loss** : cross-entropy.

**TTA inference** : 4 rotations × {original, flip horizontal} = 8 forward passes, moyenne des softmax.

### 2.5. Auto-labélisation des crops réels (`scripts/autolabel_train.py`)

Idée : sur les 81 images train, on a le GT (liste des cartes par joueur). On extrait les crops par détection heuristique, on les assigne à leur joueur par angle, puis on matche chaque crop à une carte GT du joueur par **couleur dominante** la plus proche.

→ 389 crops réels validés, écrits dans `data/real_crops/<label>/<id>.png`.

**Pourquoi c'est crucial** : le classifieur entraîné uniquement sur les templates synthétiques overfittait aux conditions parfaites (templates 200×300 propres, fond uniforme). Les vrais crops ont du bg leakage, du flou, des reflets — leur ajout au training fait passer le F1 de 0.42 → 0.69.

### 2.6. Inférence globale (`src/inference.py`, mode **hybride**)

```python
predict_scene(image, image_id, classifier, hybrid=True, detector=detector, use_tta=True)
```

1. Détecter le token → exclure sa zone.
2. **Hybride** :
   - heuristique → `heur_cards` ;
   - détecteur appris → `model_cards` ;
   - **garde l'heuristique** pour les cartes en zone centrale (CenterAcc 0.901 vs 0.728 du détecteur appris) ;
   - **garde le détecteur appris** pour les cartes hors centre (F1 0.753 vs 0.671).
3. Classifier les crops (TTA 8×).
4. Filtrer par `confidence ≥ 0.40`.
5. Assigner les cartes aux joueurs par angle depuis le centre.
6. Assigner le token au joueur dont la carte la plus proche du token appartient.

---

## 3. Historique des itérations

| Version | Description | Score (train) | Kaggle LB |
|---|---|---|---|
| v1 baseline | UnoCNN, templates synthétiques seuls | 0.382 | — |
| v1 + fix token | Token noir/jaune adaptatif HSV | 0.420 | — |
| v2 bg_fringe | + augmentation fond synthétique autour des cartes | 0.418 (régression) | — |
| v3 non_card | + classe « non_card » sur bg patches | 0.395 (régression) | — |
| **v4 real-mix** | Fine-tune v1 sur 389 crops réels auto-labellisés | **0.696** | **0.539** |
| v4 + TTA | + 8× TTA inference | 0.696 | — |
| v5a learned-detector | Détecteur appris seul (après fix focal loss) | 0.747 | TBD |
| **v5b hybride** | Heuristique pour le centre + détecteur pour les joueurs | **0.764** | TBD |

### Ce qui a marché

1. **Auto-labélisation des crops réels** → +0.27 sur le score : les vrais crops avec bg leakage matchent la distribution de test.
2. **Détecteur appris** sur 626 bboxes annotées manuellement → +0.05 sur F1.
3. **Mode hybride** : cumule le meilleur de CenterAcc (heuristique 0.901) et F1 (détecteur appris 0.753).
4. **Token detection adaptatif** : threshold dynamique selon le V médian de l'image.

### Ce qui n'a pas marché

1. **bg_fringe augmentation** : régression de 0.420 → 0.418. Le fond synthétique ne reproduit pas les vraies leakages, et le modèle a appris « tout fond bruité = carte » (faux positifs sur feuillage).
2. **Classe non_card** : régression à 0.395. Déséquilibre (1215 bg patches vs 54 templates × augs) → le modèle classe trop de vraies cartes comme « non_card ».
3. **Premier training du détecteur** : 80 epochs avec `obj_loss = 0.0000` du début à la fin. Cause : `pos_mask = (target == 1.0)` ne sélectionnait quasiment aucun pixel parce que la heatmap gaussienne discrète a un max < 1.0 quand le centre tombe entre deux cellules. **Fix** : `obj_target[cy, cx] = 1.0` explicitement au pixel central.
4. **Confidence threshold sweep** : optimum à 0.40. Plus haut → ActiveAcc et F1 chutent (classifieur sort des conf moyennes sur les vraies cartes).

---

## 4. Code & scripts

### `src/`

| Fichier | Rôle |
|---|---|
| `config.py` | 54 classes UNO, paths, géométrie joueurs |
| `templates.py` | extraction des 54 templates depuis reference_images |
| `augmentation.py` | augmentations à la volée du dataset synthétique |
| `dataset.py` | `UnoTemplateDataset` mixant templates + crops réels |
| `model.py` | `UnoCNN` (ResNet-18-light, 11.20M params) |
| `detection.py` | détection heuristique HSV + ovales blancs + token |
| `detector.py` | détecteur appris CenterNet (0.53M params) |
| `detector_dataset.py` | dataset pour entraîner le détecteur |
| `detector_inference.py` | runtime + NMS du détecteur appris |
| `inference.py` | `predict_scene` (pipeline complet, mode hybride) |

### `scripts/`

| Script | Usage |
|---|---|
| `download_data.py` | télécharge le dataset via kagglehub |
| `extract_templates.py` | génère `data/card_templates/` |
| `autolabel_train.py` | génère `data/real_crops/` (389 crops auto-labellisés) |
| `generate_yolo_preannots.py` | pré-annotations bbox pour LabelImg |
| `train_classifier.py` | entraîne UnoCNN, flags `--init-from`, `--real-crop-prob` |
| `train_detector.py` | entraîne le détecteur CenterNet |
| `eval_on_train.py` | éval end-to-end, flags `--tta`, `--no-detector`, `--hybrid` |
| `viz_pipeline.py` | visualise zones joueurs + cartes + token |
| `check_model.py` / `check_detector.py` | comptage des paramètres |

### Entry point

```bash
# Évaluation locale sur les 81 images train
python scripts/eval_on_train.py --tta --hybrid

# Génération de la submission Kaggle
python main.py --tta --hybrid --output outputs/submissions/submission.csv
```

---

## 5. Annotation des bboxes (LabelImg)

626 bboxes annotées à la main sur les 81 images train (1 classe « card »).

**Patches LabelImg requis pour Python 3.12 + PyQt5 récent** (déjà appliqués) :

1. `…/site-packages/libs/canvas.py` lignes 526, 530-531 : wrap `int(…)` sur tous les `.x()`, `.y()`, `.width()`, `.height()` dans `drawRect`/`drawLine`.
2. `…/site-packages/labelImg/labelImg.py` lignes 965, 1025-1026 : wrap `int(…)` sur les `setValue(…)`.

Workflow :
- format YOLO (bouton de gauche) ;
- Open Dir : `data/train_images` ;
- Change Save Dir : `data/yolo_annotations` ;
- Auto Save activé ;
- Raccourcis : `w` créer bbox, `d` suivant, `a` précédent.

---

## 6. Reproduire les résultats

```bash
# 1. Setup (requirements.txt)
pip install -r requirements.txt

# 2. Télécharger le dataset Kaggle
python scripts/download_data.py     # nécessite ~/.kaggle/kaggle.json

# 3. Extraire les templates
python scripts/extract_templates.py

# 4. Auto-labéliser les crops réels (suppose les annotations YOLO présentes)
python scripts/autolabel_train.py

# 5. Entraîner le classifieur (synthétique + crops réels)
python scripts/train_classifier.py --epochs 60 --real-crop-prob 0.5

# 6. Entraîner le détecteur (suppose data/yolo_annotations/ présent)
python scripts/train_detector.py --epochs 80 --samples-per-epoch 1000

# 7. Éval locale
python scripts/eval_on_train.py --tta --hybrid

# 8. Submission Kaggle
python main.py --tta --hybrid
```

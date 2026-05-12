# UNO Vision — IAPR 2026 Final Project

Pipeline d'analyse d'images de parties de UNO. Pour chaque image, on prédit :
- la carte centrale,
- le joueur actif (parmi p1/p2/p3/p4),
- les cartes en main de chaque joueur.

## Approche

Pipeline **hybride** :
1. Détection / segmentation des cartes par traitement d'image classique (seuillage, morpho, contours).
2. Classification de chaque crop de carte par un petit CNN entraîné from scratch (≤ 12M params, sans dataset externe ni pré-entraînement).
3. Assignation joueur ↔ carte par géométrie (position dans l'image).
4. Détection du jeton actif par couleur/forme.

## Structure

```
Projet_IAPR/
├── data/                  # dataset Kaggle (non versionné)
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── sample_submission.csv
├── src/                   # code source du pipeline
│   └── config.py          # constantes (classes UNO, géométrie joueurs)
├── notebooks/             # rapport Jupyter
├── outputs/
│   ├── models/            # poids entraînés
│   └── submissions/       # CSVs pour Kaggle
├── main.py                # produit le CSV de soumission final
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py            # génère outputs/submissions/submission.csv
```

## Contraintes du projet

- Pas de datasets externes.
- Pas de modèles pré-entraînés.
- ≤ 12M paramètres.
- Test set utilisé uniquement pour l'inférence.

## Deadline

- Code + notebook : **20 mai 2026, 23h55**.
- Présentation orale : **22 mai 2026**.

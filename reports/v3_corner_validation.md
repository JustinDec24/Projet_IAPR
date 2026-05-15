# Phase 1 — Validation empirique de l'approche "coin"

**Test set** : 237 coins (35%×35% haut-gauche) extraits des crops réels labellisés, 53 classes.

**Classifieur testé** : `classifier_v4_realmix.pt` (11.2M, entraîné sur cartes ENTIÈRES → les coins sont out-of-distribution pour lui).

## Résultats

| Métrique | Accuracy |
|---|---|
| Coin, sans TTA | 0.063 |
| Coin, avec TTA 8× | 0.059 |
| Couleur seule (TTA) | 0.540 |

## Top confusions

| GT | Prédit | Count |
|---|---|---|
| g_2 | g_6 | 6 |
| g_4 | g_6 | 6 |
| y_7 | wild | 6 |
| g_7 | wild | 5 |
| r_7 | r_9 | 5 |
| y_8 | wild | 5 |
| draw_4 | wild | 4 |
| g_8 | g_6 | 4 |
| r_4 | r_9 | 4 |
| r_draw_2 | r_9 | 4 |
| y_3 | y_6 | 4 |
| b_1 | wild | 3 |

## Verdict : **NO-GO**

Accuracy trop faible (<=60%). Retour à l'approche OBB.

### Interprétation

- L'accuracy *full-card classifier sur coins* est une **borne basse** (domain shift : il n'a jamais vu de coins seuls).
- L'accuracy **couleur seule = 0.540** indique si le coin porte au moins le signal couleur (toujours vrai par design UNO).
- Le **corner-classifier dédié** (Phase 4), entraîné spécifiquement sur des coins, devrait nettement dépasser ces chiffres.
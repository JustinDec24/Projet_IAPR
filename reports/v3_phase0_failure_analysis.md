# Phase 0 — Failure Analysis du baseline 0.764

**Setup** : `classifier_v4_realmix.pt` (11.2M) + `detector_baseline.pt` (0.53M), mode hybride + TTA, 81 images train, 546 cartes GT.

## Score décomposé

| Métrique | Valeur | Erreurs |
|---|---|---|
| CenterAcc | 0.864 | 11 / 81 |
| ActiveAcc | 0.790 | 17 / 81 |
| Images parfaites | 12 % | 10 / 81 |

## Où sont les ~23.6 % d'erreurs ? — Décomposition des 182 erreurs cartes

| Catégorie | Count | % erreurs cartes | Lecture |
|---|---|---|---|
| **misclassification** (carte détectée, mauvais label) | **107** | **58.8 %** | 🔴 dominant |
| FP_detection (hallucination / sur-détection) | 52 | 28.6 % | 🟠 |
| FN_detection (carte ratée) | 18 | 9.9 % | 🟡 |
| wrong_player (bon label, mauvaise zone) | 5 | 2.7 % | ✅ négligeable |
| **Total** | **182** | sur 546 cartes GT | |

Sur-détection : 580 cartes prédites vs 546 GT (+34 net).

## Verdict — direction V3 "corner detection + corner classification"

**59 % des erreurs cartes = misclassification.** Le classifieur se trompe de label *même quand la carte est correctement localisée*. C'est exactement la faiblesse que l'approche "coin" cible :

1. **Misclassification (59 %)** → Les cartes UNO impriment couleur+valeur dans les coins (HG et BD). Un crop de coin propre 80×80 contient l'info discriminante sans le bruit du centre (ovale blanc, reflets, fond). Classifier le coin plutôt que la carte entière warpée devrait réduire ces erreurs.

2. **FP_detection (29 %)** → La déduplication géométrique V3 (2 coins compatibles à ~diagonale carte = 1 carte) filtre naturellement les hallucinations isolées (1 seul coin sans partenaire géométrique).

3. **FN_detection (10 %)** → La redondance des 2 coins par carte aide à l'occlusion : une carte partiellement cachée garde souvent ≥1 coin visible (les piles décalent en diagonale, exposant le coin).

4. **wrong_player (2.7 %)** → Déjà quasi-parfait. L'assignation par zones fixes V3 ne dégradera pas. **Ne pas investir ici.**

5. **ActiveAcc 0.790** → confusions surtout `p3→p2`, `p2→p4` (token raté ou mal zoné). Gain max +0.02 sur le score. Token HSV V3 + zones fixes peuvent aider, mais 2nd ordre.

6. **CenterAcc 0.864** → 11 erreurs, surtout misclassification (`r_8→r_0`, `g_7→g_reverse`) et 2 EMPTY. Même cause racine que #1.

### Conclusion

Le levier dominant est la **qualité de classification** (59 % des erreurs). L'approche corner-classification attaque directement la cause racine. La déduplication géométrique adresse le 2nd levier (FP 29 %). **GO conceptuel pour V3** — à confirmer empiriquement en Phase 1 (le classifieur 11.2M atteint-il >80 % sur des crops de coins ?).

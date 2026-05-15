# V3 — Corner Detection + Corner Classification : rapport final

## 1. Hypothèse

Les cartes UNO impriment couleur+valeur dans 2 coins diagonalement opposés.
Détecter+classifier ces coins devrait : (a) résoudre l'occlusion (1 coin
suffit), (b) attaquer la misclassification (59 % des erreurs V2 d'après la
Phase 0), via un crop coin propre sans le bruit du centre.

## 2. Pipeline construit

| Composant | Détail | Params |
|---|---|---|
| Corner detector | CenterNet-like, heatmap+offsets, [40,80,160,320] | 3.19M |
| Corner classifier | CNN 4 blocs [48,96,160,224], crop 80×80, 54 classes | 1.30M |
| Token / zones / dédup | classique HSV + géométrie V2 | 0 |
| **Total** | | **4.49M** (<12M, large marge) |

Dataset synthétique : 5000 scènes, 67 731 coins annotés (`class x y occ
crop_size`), fonds extraits des 81 vraies images, occlusion calculée par
owner-map. Coins réels : extraction *rotation-aware* (minAreaRect couleur)
depuis les 626 bboxes axis-aligned.

## 3. Résultats par composant

| Composant | Métrique | Valeur |
|---|---|---|
| Corner detector (Phase A synth) | P / R val | 0.999 / 0.999 |
| Corner detector (Phase B réel) | P / R / F1 val réel | 0.983 / 0.987 / **0.985** |
| Corner classifier | acc train (synth+tpl) | 0.999 |
| Corner classifier | **acc réel (real_crops)** | **0.759** |

## 4. Résultat end-to-end

| Variante | CenterAcc | ActiveAcc | F1 | Score |
|---|---|---|---|---|
| V3 zones fixes | 0.099 | 0.741 | 0.412 | 0.413 |
| V3 + assignation géométrique V2 | 0.074 | 0.741 | 0.407 | 0.407 |
| V3 + dédup géométrique pure | — | 0.741 | 0.384 | 0.391 |
| **V2 (référence)** | **0.85+** | **0.80** | **0.79** | **0.787** |

## 5. Pourquoi V3 échoue : analyse

Le détecteur de coins est **excellent** (F1 0.985 réel). Le mur est ailleurs :

1. **Gap synthétique→réel du corner classifier** : 0.999 train → **0.759
   réel**. Les crops de coins synthétiques (cartes templates collées) ne
   reproduisent pas l'apparence des vrais coins (usure, reflets, flou, bord
   réel). 24 % d'erreur par coin.

2. **Multiplication d'erreurs dans un pipeline multi-étapes** :
   `détection 0.99 × classif/coin 0.76 × (accord 2 coins)` → précision carte
   nette ≪ celle du classifieur full-card V2 (qui voit toute la carte d'un
   coup, contexte complet).

3. **Dilemme du pairing** :
   - Pairing *avec accord de label* → P(2 coins même label correct) ≈
     0.76² ≈ 0.58 → la plupart des cartes restent en 1-coin, mal localisées
     (centre = position d'un coin) → assignation zone cassée → CenterAcc 0.07.
   - Pairing *géométrique pur* → mé-pairing de coins de cartes voisines à
     distance de diagonale → labels incohérents → F1 0.38.

4. **CenterAcc 0.07-0.10** : conséquence directe de (3). La carte centrale,
   pourtant isolée et facile en V2 (0.86), est mal positionnée car composée
   d'un seul coin → tombe dans une mauvaise zone.

## 6. Leçon (valeur pédagogique)

> Un sous-composant excellent (corner detector 0.99) ne garantit pas un
> pipeline performant si un autre maillon (corner classifier 0.76 réel à
> cause du gap synthétique) et la **composition des incertitudes** dominent.
> Le classifieur full-card V2 voit le contexte entier en une passe : moins
> de maillons = moins de multiplication d'erreurs.

Le risque avait été explicitement assumé en Phase 1 : le test "classifieur
V2 sur coins" donnait 5.9 % (NO-GO strict), choix *option C — GO direct sur
preuve visuelle*. La preuve visuelle (coins lisibles par un humain) était
correcte mais ne garantissait pas qu'un CNN entraîné sur synthétique
généralise au réel sur ces petits crops.

## 7. Décision

**Submission = V2 (Score 0.787, branche `v2`, `submission_v2.csv`).**

V3 est conservé sur la branche `v3` comme étude rigoureuse (détecteur de
coins réutilisable à 0.985, dataset synthétique, analyse d'échec quantifiée).
Pistes non explorées faute de temps/ROI :
- Combler le gap : extraire ~1100 vrais coins (rotation-aware sur les 81
  images) pour ré-entraîner le classifier avec bien plus de réel.
- Pipeline hybride : détecteur de coins pour la *localisation* (0.99) +
  classifieur full-card pour le *label* (réintroduit le coût params V2).

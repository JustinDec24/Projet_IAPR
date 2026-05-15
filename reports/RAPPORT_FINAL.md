# UNO Vision — Rapport de projet IAPR 2026

**Détection et analyse de scènes de jeu UNO** — du baseline classique (0.764) à
un pipeline hybride à **0.850** via trois versions successives.

---

## 1. Contexte et contraintes

Pour chaque image d'une partie de UNO (4000×2662 px, fonds blancs ou tropicaux
bruités), on prédit :
- la **carte centrale** posée au milieu,
- le **joueur actif** (p1–p4, désigné par un jeton),
- les **cartes en main** de chacun des 4 joueurs.

**Métrique** : `Score = 0.1·CenterAcc + 0.1·ActiveAcc + 0.8·F1` (F1 multi-ensemble
sur les cartes des joueurs).

**Contraintes strictes** : ≤ 12M paramètres au total (tous modèles confondus),
**aucun modèle pré-entraîné**, **aucun dataset externe**, **aucun poids
ImageNet**. 81 images d'entraînement annotées (CSV), 626 bounding boxes tracées
manuellement, 54 classes de cartes, images de référence des 54 cartes.

**Baselines de référence** : classique 0.570, Deep Learning 0.647.

---

## 2. Vue d'ensemble — le fil rouge des trois versions

Le projet s'est construit en **trois versions**, chacune répondant à la
limite de la précédente :

| Version | Idée directrice | Score | Apport décisif |
|---|---|---|---|
| **V1 — Baseline hybride** | détection classique (HSV) + CNN classifieur from scratch + auto-labeling | **0.764** | prouve la viabilité hybride classique/DL |
| **V2 — Industrialisation** | données synthétiques + détecteur appris + distillation | **0.787** | détection apprise, budget params maîtrisé |
| **V3 — Corner detection** | détecter/classifier les *coins* des cartes | 0.41 (direct) | **échec en direct, mais son détecteur de coins débloque tout** |
| **Convergence (V2 final)** | V3 → auto-labeling itératif + ensemble centre | **0.850** | +0.086 vs baseline |

Le récit central : **V3 n'a jamais battu V2 en compétition directe, mais c'est
son détecteur de coins — sous-produit d'une exploration "ratée" — qui a permis
les deux gains finaux qui ont porté V2 de 0.787 à 0.850.**

---

## 3. Version 1 — Baseline hybride classique + Deep Learning

### 3.1 Architecture

Pipeline en quatre étages, sans aucun pré-entraînement :

1. **Détection des cartes (classique, 0 param)** — deux stratégies fusionnées :
   - *Saturation* (fond blanc) : flou gaussien sur le canal S de HSV →
     seuillage → contours → `minAreaRect`. Pour les piles de cartes empilées,
     **split par couleur** (k-means couleur si le blob contient ≥2 couleurs
     UNO distinctes) puis split géométrique en dernier recours.
   - *Ovales blancs* (fond bruité) : la saturation est inutilisable ; on
     exploite l'**ovale blanc central** de chaque carte, qui forme un trou
     *enclosed* dans le masque saturé.
2. **Classification (CNN UnoCNN, 11.2M params)** — ResNet-18-light entraîné
   *from scratch* : stem 3×3, 4 stages de 2 BasicBlocks (64/128/256/512),
   GAP + Linear(512, 54). Input crop 144×96 (downscalé de 200×300).
3. **Détection du jeton (classique)** — seuil V adaptatif (`médiane V − 80`) :
   fond clair → jeton noir, fond bruité → jeton jaune.
4. **Assignation géométrique** — distance au centre image (< 18 % diag → centre)
   puis angle pour p1–p4 ; jeton attribué au joueur de la carte la plus proche.

### 3.2 Technique-clé : l'auto-labeling

Le classifieur entraîné sur **templates synthétiques seuls** plafonnait à
0.42 (overfit aux templates parfaits, domain gap avec les vraies cartes
usées/floues). Solution : **auto-labeling** des 81 images d'entraînement —
on détecte les cartes, on les assigne aux joueurs par géométrie, puis on
matche chaque crop au label GT du joueur **par couleur dominante**. Cela
produit 389 crops réels labellisés. Le fine-tuning du classifieur dessus
fait bondir le score **0.42 → 0.69 train / 0.539 Kaggle** (+0.27 sur le F1).

> **Leçon V1** : sur ce dataset, le levier dominant n'est pas l'architecture
> mais l'**adéquation des données d'entraînement à la distribution réelle**.
> L'auto-labeling itératif devient le fil conducteur technique du projet.

### 3.3 Limites de V1

- Détection 100 % heuristique → fragile sur fonds tropicaux (faux positifs
  feuillage) et sur cartes occluses dans les piles.
- Classifieur 11.2M : tout le budget params consommé, pas de marge pour un
  détecteur appris.
- Score 0.764 : CenterAcc 0.901, ActiveAcc 0.716, F1 ~0.74.

---

## 4. Version 2 — Industrialisation : synthétique, détecteur appris, distillation

### 4.1 Analyse d'échec quantifiée (étape préalable obligatoire)

Avant toute refonte, **décomposition des 182 erreurs cartes** du baseline en
6 catégories :

| Catégorie | % des erreurs |
|---|---|
| **Misclassification** (carte localisée, mauvais label) | **58.8 %** |
| FP_detection (hallucination) | 28.6 % |
| FN_detection (carte ratée) | 9.9 % |
| wrong_player (bon label, mauvaise zone) | 2.7 % |

**Conclusion chiffrée** : 88 % des erreurs sont détection+classification,
l'assignation joueur est déjà quasi-parfaite (ne pas y toucher). Toute
l'énergie doit aller à la qualité détection/classification.

### 4.2 Génération de données synthétiques (assets internes uniquement)

Conforme à la règle "pas de données externes" : composition à partir des
**54 templates** + **patches de fond** extraits des 81 images réelles
(zones sans carte, via l'inverse des bboxes + dilatation). **5000 scènes**
générées : 4–13 cartes, rotations ±20°, empilements diagonaux 30–50 px,
ombres douces, occlusions calculées par *owner-map*, annotations exactes
par construction.

### 4.3 Détecteur appris CenterNet (et un bug instructif)

Détecteur anchor-free style CenterNet (heatmap d'objectness + régression
bbox), channels scalés de [16,32,64,96] (0.53M) à **[40,80,160,320]
(4.77M)**, entraîné en **2 phases** : pré-entraînement synth 80 % + réel
20 %, puis fine-tune réel.

> **Bug trouvé et corrigé** : la focal loss `pos_mask = (target == 1.0)` ne
> sélectionnait *jamais* de pixel positif — le max de la heatmap gaussienne
> discrète est < 1.0 quand le centre tombe entre deux cellules. Le détecteur
> n'apprenait que "tout zéro". Fix : forcer `target = 1.0` au pixel central.

> **Tentative écartée** : un détecteur OBB + FPN multi-niveaux. La régression
> bbox s'effondrait (sigmoid saturée → valeur fixe 0.17×0.26 quelle que soit
> la carte). Diagnostiqué, documenté, abandonné — leçon sur la composition
> loss/activation.

### 4.4 Distillation du classifieur (libérer du budget params)

Le classifieur 11.2M + détecteur 4.77M = 15.97M > 12M. **Distillation
knowledge** : teacher 11.2M → student **6.31M** (channels [48,96,192,384]),
loss `0.3·CE + 0.7·KL(student/T ‖ teacher/T)·T²`, T=4. val_acc student =
**0.925**. Budget final : **4.77M + 6.31M = 11.08M < 12M** ✓.

### 4.5 Gains classiques 0-param

- **Token fallback double-essai** : on tente noir *et* jaune (au lieu de
  switcher sur la médiane V) → ActiveAcc 0.716 → **0.790**.
- **Seuil de confiance 0.40 → 0.50** : +0.002.
- **Carte centrale hors filtre confiance** (unique, EMPTY toujours faux) :
  +0.003.

**V2 = 0.787** (CenterAcc 0.691, ActiveAcc 0.790, F1 0.797). Total 11.08M.

---

## 5. Version 3 — L'approche "corner detection"

### 5.1 Hypothèse

Les cartes UNO impriment couleur+valeur dans **deux coins diagonalement
opposés**. Détecter+classifier ces coins devrait : (a) résoudre l'occlusion
(un seul coin visible suffit), (b) attaquer les 59 % de misclassification
(un crop de coin propre, sans le bruit de l'ovale central).

### 5.2 Démarche en 8 phases (avec check-points de validation)

- **Phase 0** — failure analysis baseline (cf. §4.1).
- **Phase 1** — validation empirique : le classifieur V2 (full-card) sur des
  crops de coins = **5.9 %** (NO-GO strict). *Mais* preuve visuelle : 7/8
  coins lisibles par un humain. Décision assumée : GO conditionnel (le test
  était biaisé — classifieur OOD sur coins).
- **Phase 2** — 5000 scènes synthétiques avec annotations *coins*
  (`class x y is_occluded crop_size`), occlusion via owner-map.
- **Phase 3** — détecteur de coins CenterNet **3.19M**, 2 phases
  (synth+réel puis fine-tune). Coins réels extraits des bboxes via
  **rotation-aware** (`minAreaRect` couleur, bien plus précis que l'approx
  axis-aligned). Résultat : **F1 réel 0.985 (P=0.98, R=0.99)** — excellent.
- **Phase 4** — classifieur de coins **1.30M**, crop 80×80. Résultat :
  train 0.999 mais **val réel 0.759** — gros gap synthétique→réel.
- **Phases 5–6** — zones fixes, token HSV, déduplication géométrique,
  `predict_scene_v3`. Total V3 : **4.49M params**.
- **Phase 7** — éval end-to-end + ablations.
- **Phase 8** — documentation.

### 5.3 Pourquoi V3 échoue en direct

| Variante | Score |
|---|---|
| V3 corner pur | 0.41 |
| V3 dédup géométrique | 0.39 |
| Hybride rotated-rect | 0.616 |
| Hybride axis-aligned + token-fix | **0.741** (meilleur) |
| **V2 (référence)** | **0.787** |

Trois causes racines documentées :
1. **Gap synthétique→réel** du corner-classifier (0.999 → 0.759) : les coins
   synthétiques ne reproduisent pas usure/reflets/flou.
2. **Multiplication d'erreurs** : `det 0.99 × clf/coin 0.76 × accord-2-coins`
   ≪ classifieur full-card V2 (contexte complet, une passe).
3. **Dilemme du pairing** : accord-label → cartes 1-coin mal localisées
   (CenterAcc 0.07) ; géométrique pur → mé-pairing voisins (F1 0.38).

> **Leçon V3** : un sous-composant excellent (détecteur 0.99) ne sauve pas
> un pipeline si la composition des incertitudes et un domain-gap dominent.

---

## 6. La convergence — comment V3 débloque V2 (0.787 → 0.850)

V3 a échoué comme pipeline, **mais son détecteur de coins (R=0.99,
robuste occlusion) est un actif réutilisable**. Deux exploitations :

### 6.1 Auto-labeling itératif via le corner-detector (0.787 → 0.820)

Le pattern V1 ("meilleur détecteur → meilleures données → meilleur
classifieur") rejoué : le corner-detector v3 localise les cartes des 81
images → **406 crops réels labellisés, 54/54 classes** (vs 237/53 pour V1,
+71 %, et la classe `wild` enfin couverte — le détecteur catche les cartes
occluses que l'heuristique ratait). Fine-tune du student dessus :

| | V2 (0.787) | + student v3 |
|---|---|---|
| F1 | 0.797 | **0.844** (+0.047) |
| ActiveAcc | 0.790 | 0.802 |
| CenterAcc | 0.691 | 0.642 |
| **Score** | **0.787** | **0.820** |

*Variantes testées et écartées* : crops combinés 237+406 (0.814, dilution) ;
template-matching NCC pour le centre (0.21, non rotation-invariant).

### 6.2 Ensemble 0-param de la carte centrale (0.820 → 0.850)

Constat : en mode hybride **deux détecteurs tournent déjà** (heuristique
pour le centre, appris 4.77M pour les joueurs) ; on n'utilisait que la vue
heuristique du centre. **Ensemble 0 paramètre** : garder les *deux* vues du
centre (heuristique + détecteur appris), classifier les deux, prendre la
**plus confiante** parmi les candidats proches du centre image.

| | + student v3 (0.820) | + center ensemble |
|---|---|---|
| **CenterAcc** | 0.642 | **0.938** (+0.296) |
| ActiveAcc | 0.802 | 0.802 |
| F1 | 0.844 | 0.845 |
| **Score** | **0.820** | **0.850** |

CenterAcc dépasse même la baseline 11.2M (0.901) : le détecteur appris voit
souvent mieux la carte centrale, le max-confidence sélectionne la bonne vue.

---

## 7. Résultats finaux et ablations

### 7.1 Tableau récapitulatif (81 images train)

| Pipeline | Params | CenterAcc | ActiveAcc | F1 | **Score** |
|---|---|---|---|---|---|
| Baseline DL (réf.) | — | — | — | — | 0.647 |
| **V1** baseline hybride | 11.7M | 0.901 | 0.716 | ~0.74 | **0.764** |
| **V2** synth+distill | 11.08M | 0.691 | 0.790 | 0.797 | **0.787** |
| V3 corner (meilleur direct) | 4.49M | — | — | — | 0.741 |
| V2 + auto-label itératif | 11.08M | 0.642 | 0.802 | 0.844 | 0.820 |
| **V2 final + center ensemble** | **11.08M** | **0.938** | **0.802** | **0.845** | **0.850** |

### 7.2 Ablations (contribution de chaque technique)

| Technique retirée | Δ Score |
|---|---|
| − ensemble centre 0-param | −0.030 |
| − auto-label itératif (student v3) | −0.033 |
| − détecteur appris (retour heuristique) | ≈ −0.05 sur F1 |
| − token fallback | −0.007 (ActiveAcc) |
| − données synthétiques (détecteur) | détecteur ne converge pas |

### 7.3 Budget paramètres (contrainte 12M) ✓

Détecteur CenterNet 4.77M + classifieur student 6.31M = **11.08M < 12M**.
Composants classiques (détection heuristique, token, zones, ensemble
centre) : **0 paramètre**.

---

## 8. Analyse d'échec et interprétabilité

- **CenterAcc** : avant l'ensemble, la carte centrale était le point faible
  (student distillé moins fin que le teacher sur les digits proches :
  b_5↔b_2, r_8↔r_5). L'ensemble 2-vues a résolu sans coût params.
- **ActiveAcc 0.802** : échecs résiduels surtout token raté sur fonds
  tropicaux (jeton jaune confondu avec feuillage). Levier max +0.02.
- **F1 0.845** : erreurs résiduelles = cartes très occluses dans les piles
  (un seul bord visible) et confusions de digits proches.
- **Faux négatif méthodologique** (Phase 1 V3) : tester un classifieur
  full-card sur des coins (OOD) sous-estimait gravement le potentiel coin —
  documenté comme piège d'évaluation.

---

## 9. Conclusion

Le projet illustre une démarche d'ingénierie itérative pilotée par
l'**analyse d'échec quantifiée** plutôt que par l'intuition :

1. **V1** prouve l'hybride classique+DL et révèle le levier des données
   (auto-labeling, +0.27).
2. **V2** industrialise (synthétique légal, détecteur appris, distillation
   pour tenir le budget 12M) → 0.787, et documente deux bugs/échecs
   instructifs (focal loss, OBB).
3. **V3** explore une idée élégante (corner detection) qui *échoue comme
   pipeline* (gap synthétique, multiplication d'erreurs) mais **produit le
   détecteur de coins** qui, réinjecté dans V2 via auto-labeling itératif
   puis ensemble centre, fait passer le score de **0.787 à 0.850**.

**Score final : 0.850** (+0.086 vs baseline classique, +0.203 vs baseline
DL), avec un pipeline à 11.08M paramètres, entièrement entraîné from
scratch sur données internes. La leçon transférable : *un composant
développé pour une approche abandonnée peut être l'ingrédient décisif d'une
autre — l'exploration "ratée" a une valeur instrumentale*.

### Livrables

- `main` : baseline 0.764 (filet de sécurité)
- `v2` : **pipeline final 0.850** — `outputs/submissions/submission_v2_final.csv`
- `v3` : étude corner détection complète (8 phases, 6 ablations, notebooks)
- Code : `src/`, `src/v3/`, `scripts/`, `configs/` ; rapports `reports/` ;
  notebooks `notebooks/v3_*`

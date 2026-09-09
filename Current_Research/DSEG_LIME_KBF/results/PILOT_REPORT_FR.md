# Rapport complet du pilote DSEG-LIME — 20 patients

## Protocole réellement exécuté

- 20 patients indépendants : 10 MGMT non méthylés et 10 MGMT méthylés.
- Quatre patients par fold, soit deux de chaque classe dans chacun des cinq folds.
- Entrée du CNN : coupe KLF 2D la plus riche en voxels non nuls.
- Segmentation : coupe FLAIR du même patient et au même indice, SAM ViT-B puis DSEG.
- Configuration principale : 64 perturbations par itération, 2 itérations, `points_per_side=32`, `min_segment_area=5`, `sam_min_mask_area=16`, graine 42.
- Tous les masques de perturbation FLAIR ont été transférés vers KLF avant l’inférence MGMT.

## Résultat de classification sur l’échantillon pilote

- 8 prédictions correctes sur 20 (40 %), 6 prédictions de classe 0 et 14 de classe 1.
- Confiance moyenne : 0,739.

Cette valeur ne constitue pas une estimation finale de la performance de Capuozzo : l’échantillon est petit et a été construit pour tester les explications dans les cinq folds, non pour reproduire l’évaluation complète de l’article.

## Segmentation et fidélité régionale

- Médiane : 31,5 masques SAM bruts puis 5 régions DSEG finales ; étendue DSEG : 2 à 32 régions.
- La région au poids DSEG maximal est aussi celle dont la suppression produit l’effet maximal dans 16/20 cas (80 %).
- Sa suppression diminue la probabilité de la classe expliquée dans 20/20 cas.
- Baisse moyenne : 0,134 ; baisse médiane : 0,059.
- Corrélation de Spearman poids/effet : médiane 0,949 ; moyenne 0,814.

## Contrôle aléatoire à taille utile égale

Pour chaque patient, 100 contrôles suppriment aléatoirement exactement le même nombre de voxels KLF non nuls que la région DSEG principale.

- La région DSEG dépasse la baisse aléatoire moyenne dans 19/20 cas (95 %).
- Elle est supérieure au contrôle au seuil empirique `p <= 0,05` dans 13/20 cas (65 %).
- La valeur empirique médiane est `p = 0,0099`.

Ce contrôle porte sur des voxels aléatoires non contigus. Il neutralise la quantité de signal supprimée, mais pas la géométrie régionale ; il faut donc le présenter comme un contrôle négatif, pas comme une preuve anatomique.

## Courbes progressives de suppression et d’insertion

Les régions sont ajoutées ou retirées dans l’ordre décroissant de leur poids DSEG, puis comparées à 50 permutations aléatoires des mêmes régions.

- AUC de suppression meilleure que l’ordre aléatoire dans 16/20 cas (80 %).
- AUC d’insertion meilleure que l’ordre aléatoire dans 17/20 cas (85 %).
- Gain médian d’AUC : 0,0297 pour la suppression et 0,0319 pour l’insertion.

## Stabilité sur six patients et trois graines

- La segmentation est identique dans toutes les comparaisons (`ARI = 1`), car SAM/DSEG est déterministe ici.
- Corrélation moyenne des cartes de poids par patient : de 0,720 à 0,999.
- Dice moyen des masques positifs : de 0,669 à 1,000.
- Le patient 00012 est le cas le moins stable ; il possède 29 régions, ce qui confirme que 64 perturbations sont parfois insuffisantes.

## Sensibilité au nombre de perturbations

Quatre profils représentatifs ont été recalculés à 64, 128 et 256 perturbations par itération.

| Comparaison | Corrélation moyenne des poids | Dice moyen positif | Accord région principale |
|---|---:|---:|---:|
| 64 vs 128 | 0,954 | 0,818 | 50 % |
| 64 vs 256 | 0,936 | 0,916 | 50 % |
| 128 vs 256 | 0,942 | 0,832 | 100 % |

La segmentation reste identique (`ARI = 1`). L’accord parfait de la région principale entre 128 et 256, contre 50 % lorsque 64 est impliqué, justifie l’emploi d’au moins 128 perturbations pour les cas à forte granularité.

## Comparaison DSEG–KBF

Le masque KBF observable sur la coupe est défini par `KLF != 0`, puisque KLF conserve le signal FLAIR uniquement aux positions retenues par le filtre de connaissances.

- Le rappel médian du masque KBF par les régions DSEG de poids positif est 0,865.
- La précision spatiale médiane n’est que 0,019 et le Dice médian 0,038.

Ce faible Dice ne signifie pas que DSEG échoue : ses régions couvrent toute l’image FLAIR alors que KBF est très parcimonieux. Le résultat utile est le fort rappel : l’essentiel du signal KBF se situe dans des régions favorables à la classe expliquée, mais ces régions sont beaucoup plus larges que KBF. La prochaine contribution doit donc agir à l’intérieur du support KBF, et non remplacer KBF par le masque DSEG brut.

## Décision scientifique

Le pilote franchit le seuil « continuer » : la fidélité est positive, dépasse des contrôles aléatoires et demeure globalement stable. DSEG-LIME peut servir d’outil de diagnostic de KBF. Il n’est toutefois pas encore une contribution finale.

La contribution proposée pour l’étape suivante est un **KBF repondéré par l’importance DSEG**, construit uniquement sur les données d’entraînement de chaque fold. Les voxels KBF appartenant aux régions DSEG positives sont conservés/renforcés, tandis que ceux des régions négatives sont atténués. La comparaison finale doit opposer KBF original et KBF repondéré avec exactement le même CNN, les mêmes folds et les mêmes métriques MGMT.

## Fichiers produits

- `baseline_results.csv`, `faithfulness_cases.csv`, `faithfulness_regions.csv`
- `complete_evaluation_cases.csv`, `complete_evaluation_summary.json`, `progressive_curves.csv`
- `stability_pairs.csv`, `stability_summary.csv`
- `sample_sensitivity_pairs.csv`, `sample_sensitivity_summary.csv`
- `pilot_complete_summary.png`

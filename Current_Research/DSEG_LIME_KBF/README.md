# DSEG-LIME and KBF for MGMT prediction

## Research question

Can locally faithful DSEG-LIME explanations identify useful regions within a Knowledge-Based Filter and guide an improvement of MGMT methylation prediction?

## Implemented method

The CNN receives the 2D KLF slice with the largest non-zero support. SAM ViT-B and DSEG segment the FLAIR slice at the same index. DSEG-LIME perturbs those regions, transfers the masks to KLF and estimates each region's contribution to the CNN class probability.

The explanation label is the model class being explained, not a tumour segmentation label. MGMT status remains the classification ground truth.

## Twenty-patient pilot

The pilot contains 20 independent cases: 10 MGMT non-methylated and 10 methylated, with four cases from each of the five folds.

| Test | Result |
|---|---:|
| Baseline predictions correct | 8/20 (40%) |
| Top region also produced maximum deletion effect | 16/20 (80%) |
| Top-region deletion reduced class probability | 20/20 (100%) |
| Median weight/deletion Spearman correlation | 0.949 |
| Top DSEG region beat same-size random mean | 19/20 (95%) |
| Empirical random-control p <= 0.05 | 13/20 (65%) |
| DSEG deletion ordering beat random | 16/20 (80%) |
| DSEG insertion ordering beat random | 17/20 (85%) |

The 40% pilot accuracy is not a final estimate of Capuozzo performance: the sample was selected to validate explanations across classes and folds.

## Stability and KBF agreement

Stability was measured on six patients with three seeds. Segmentation was deterministic (`ARI=1`); mean weight-map correlations ranged from 0.720 to 0.999 and positive-mask Dice from 0.669 to 1.000.

On four representative patients, the top region agreed in 100% of 128-vs-256 comparisons, versus 50% when 64 perturbations were involved. At least 128 perturbations are preferred for granular cases.

Positive DSEG regions recovered a median 86.5% of KBF support, but median precision was 1.9% and Dice 3.8%. DSEG regions are broader than sparse KBF voxels, so raw DSEG masks should not replace KBF.

## Fold-0 sentinel

585 training explanations were generated at 128 perturbations x 2 iterations. Four variants were compared on 117 held-out fold-0 patients.

| Variant | Accuracy | F1 | ROC AUC |
|---|---:|---:|---:|
| Original KBF | 0.4872 | 0.5238 | 0.5070 |
| Signed DSEG, alpha=0.25 | 0.5470 | 0.5891 | 0.5053 |
| Signed DSEG, alpha=0.50 | 0.4701 | 0.4833 | 0.5023 |
| Signed DSEG, alpha=1.00 | 0.5299 | 0.5378 | 0.4944 |

At alpha=0.25, nine baseline errors were corrected and two correct predictions degraded; exact McNemar p=0.0654. AUC did not improve, so predictive improvement cannot yet be claimed.

## Scientific conclusion and next gate

DSEG-LIME passed local faithfulness, random-control and practical stability tests. Direct signed KBF reweighting did not improve fold-0 AUC. Next:

1. repeat with the exact Capuozzo protocol, particularly `batch_size=1`;
2. test area-normalized reweighting and DSEG-guided slice selection;
3. compare mechanisms on fold 0 with fixed seeds and identical CNN settings;
4. run all five folds only after a mechanism shows a defensible benefit.

## Contents

- `code/`: experiment scripts and the annotated Jupyter notebook;
- `results/`: compact tables and reports;
- `figures/`: aggregate summaries and de-identified examples.

No source MRI, private clinical record, full explanation tensor or checkpoint is included.

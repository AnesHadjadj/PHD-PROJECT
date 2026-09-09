# Interpretable Deep Learning for MGMT Prediction

Doctoral research repository of **Anes Hadjadj** on interpretable deep learning for glioblastoma MRI analysis and non-invasive prediction of **MGMT promoter methylation**.

## Research objective

Can reliable local explanations improve the Knowledge-Based Filtering (KBF) stage while preserving or improving MGMT prediction? The current pipeline combines FLAIR/T1w MRI, Capuozzo et al.'s KBF, a 2D CNN, DSEG-LIME explanations based on SAM regions, and objective faithfulness and stability tests.

## Current scientific status

| Milestone | Status | Main result |
|---|---:|---|
| Capuozzo pipeline reproduction | Completed | KBF and CNN inference reproduced locally |
| Balanced DSEG-LIME pilot | Completed | 20 patients, 10 per MGMT class |
| Explanation faithfulness | Completed | Top DSEG region caused the largest deletion effect in 16/20 cases |
| Same-size random control | Completed | DSEG beat the random mean in 19/20 cases |
| Stability analysis | Completed | 6 patients across 3 seeds |
| Perturbation sensitivity | Completed | 128 vs 256 agreed on the top region in all 4 tested cases |
| Fold-0 explanations | Completed | 585/585 training explanations generated |
| Fold-0 sentinel retraining | Completed | Accuracy/F1 increased at alpha=0.25; AUC did not improve |
| Exact-protocol confirmation | Pending | Repeat with exact Capuozzo training settings |
| Five-fold final evaluation | Pending | After selecting a defensible integration mechanism |

The evidence supports DSEG-LIME as a **faithful diagnostic tool for KBF**, but not yet as proof that direct explanation-based reweighting improves MGMT prediction. Explanation fidelity and predictive utility are distinct properties.

[Detailed experiment, results and reproducibility notes](./Current_Research/DSEG_LIME_KBF/)

## Repository structure

- [`Capuozzo/`](./Capuozzo/): reproduced baseline code and historical experimental material.
- [`Current_Research/`](./Current_Research/): active contribution and validated intermediate results.
- [`My Documents/`](./My%20Documents/): publication and academic documents.

Patient MRI volumes, private clinical data, local checkpoints and full generated tensors are intentionally excluded.

## Publication

**Deep Learning-Based MGMT Promoter Methylation Prediction: A Comparative Interpretability-Oriented Review**

2026 International Conference on Cyber Management and Engineering (CyMaEn'26) — **Accepted**

[Conference paper](./My%20Documents/IEEE_CyMaEn_Review_Conference_Paper.pdf)

## Important limitation

The 20-patient pilot validates explanation behavior; it does not estimate final classifier performance. The fold-0 sentinel used a common `batch_size=32`, rather than the historical Capuozzo `batch_size=1`. Comparisons between sentinel variants are fair, but exact baseline equivalence remains to be confirmed.

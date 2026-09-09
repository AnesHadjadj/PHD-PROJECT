# Code and Jupyter workflow

The notebook provides the readable entry point; the Python files contain the complete experiment and evaluation logic used during the pilot and fold-0 sentinel.

## Local prerequisites

- the Capuozzo repository and trained fold checkpoints;
- KLF and FLAIR MAT volumes generated from the authorized dataset;
- DSEG-LIME and Segment Anything (SAM ViT-B);
- Python environment matching the imports in the scripts.

Some scripts preserve the experiment machine's default `D:\PHD\...` dataset paths for provenance. Override those paths with command-line arguments where available, or adapt the constants to your authorized local data location. No dataset is downloaded automatically.

## Suggested reading order

1. `dseg_lime_mri_probe.ipynb`
2. `dseg_lime_mri_probe.py`
3. `evaluate_dseg_faithfulness.py`
4. `evaluate_dseg_complete.py`
5. `evaluate_dseg_stability.py`
6. `evaluate_sample_sensitivity.py`
7. `run_fold0_sentinel_dseg.py`
8. `train_fold0_sentinel.py`

The notebook outputs were cleared before publication to avoid embedding local paths and bulky execution traces. Aggregate verified results remain under `../results/`.

"""Compare les répétitions DSEG-LIME d'un même patient."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent
PATTERN = "patient_*_pilot_p32_i2_s64_ma5_sa16_r*"


def dice(a, b):
    a, b = np.asarray(a) != 0, np.asarray(b) != 0
    denominator = int(a.sum() + b.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)


groups = {}
for folder in sorted((HERE / "outputs").glob(PATTERN)):
    result_path, npz_path = folder / "result.json", folder / "dseg_lime_result.npz"
    if result_path.exists() and npz_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        groups.setdefault(result["patient"], []).append((folder.name, np.load(npz_path)))

records = []
for patient, runs in groups.items():
    for (name_a, a), (name_b, b) in itertools.combinations(runs, 2):
        segments_a, segments_b = a["segments"], b["segments"]
        mask_a, mask_b = a["positive_mask"], b["positive_mask"]
        weight_a, weight_b = a["weight_map"].ravel(), b["weight_map"].ravel()
        if np.std(weight_a) == 0 or np.std(weight_b) == 0:
            weight_corr = np.nan
        else:
            weight_corr = float(np.corrcoef(weight_a, weight_b)[0, 1])
        records.append({
            "patient": patient,
            "run_a": name_a.rsplit("_", 1)[-1],
            "run_b": name_b.rsplit("_", 1)[-1],
            "segmentation_ari": float(adjusted_rand_score(segments_a.ravel(), segments_b.ravel())),
            "positive_mask_dice": dice(mask_a, mask_b),
            "weight_map_correlation": weight_corr,
        })

if not records:
    raise SystemExit("Pas assez de répétitions pour calculer la stabilité.")

details = pd.DataFrame(records)
summary = details.groupby("patient", as_index=False)[
    ["segmentation_ari", "positive_mask_dice", "weight_map_correlation"]
].mean()
details.to_csv(HERE / "pilot" / "stability_pairs.csv", index=False)
summary.to_csv(HERE / "pilot" / "stability_summary.csv", index=False)
print("Comparaisons par paire")
print(details.to_string(index=False))
print("\nMoyennes par patient")
print(summary.to_string(index=False))

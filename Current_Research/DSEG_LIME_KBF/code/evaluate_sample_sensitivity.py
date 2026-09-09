"""Compare DSEG-LIME à 64, 128 et 256 perturbations par itération."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent
SAMPLES = (64, 128, 256)
PATIENTS = ("00003", "00009", "00012", "00026")


def dice(a, b):
    a, b = np.asarray(a) != 0, np.asarray(b) != 0
    return float(2 * (a & b).sum() / max(int(a.sum() + b.sum()), 1))


records = []
for patient in PATIENTS:
    runs = {}
    for samples in SAMPLES:
        matches = list((HERE / "outputs").glob(
            f"patient_{patient}_fold*_pilot_p32_i2_s{samples}_ma5_sa16_r0"
        ))
        if len(matches) != 1:
            raise SystemExit(f"Résultat unique introuvable : patient={patient}, samples={samples}")
        folder = matches[0]
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        runs[samples] = (result, np.load(folder / "dseg_lime_result.npz"))
    for a, b in itertools.combinations(SAMPLES, 2):
        ra, da = runs[a]
        rb, db = runs[b]
        wa, wb = da["weight_map"].ravel(), db["weight_map"].ravel()
        corr = np.nan if np.std(wa) == 0 or np.std(wb) == 0 else float(np.corrcoef(wa, wb)[0, 1])
        records.append({
            "patient": patient, "samples_a": a, "samples_b": b,
            "segments_a": ra["segments"], "segments_b": rb["segments"],
            "segmentation_ari": float(adjusted_rand_score(da["segments"].ravel(), db["segments"].ravel())),
            "positive_mask_dice": dice(da["positive_mask"], db["positive_mask"]),
            "weight_map_correlation": corr,
            "same_top_region": int(np.argmax([
                da["weight_map"][da["segments"] == sid].mean() for sid in np.unique(da["segments"])
            ])) == int(np.argmax([
                db["weight_map"][db["segments"] == sid].mean() for sid in np.unique(db["segments"])
            ])),
        })

details = pd.DataFrame(records)
summary = details.groupby(["samples_a", "samples_b"], as_index=False).agg(
    patients=("patient", "count"),
    mean_positive_mask_dice=("positive_mask_dice", "mean"),
    mean_weight_map_correlation=("weight_map_correlation", "mean"),
    top_region_agreement_rate=("same_top_region", "mean"),
    mean_segmentation_ari=("segmentation_ari", "mean"),
)
details.to_csv(HERE / "pilot" / "sample_sensitivity_pairs.csv", index=False)
summary.to_csv(HERE / "pilot" / "sample_sensitivity_summary.csv", index=False)
print(summary.to_string(index=False))

"""Mesure la fidélité de DSEG-LIME par suppression régionale."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from baseline_2d_probe import RSNAAlternativeClassifierSingle2D, best_checkpoint, load_state_dict


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODEL_FOLDERS = [
    "fold0_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_09-40-29",
    "fold1_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_10-15-21",
    "fold2_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_10-50-28",
    "fold3_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_09-42-53",
    "fold4_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_10-18-34",
]


def class_probabilities(model, images, device):
    batch = torch.from_numpy(images[:, None].astype(np.float32)).to(device)
    chunks = []
    with torch.inference_mode():
        for local in batch.split(32):
            p1 = torch.sigmoid(model(local)).reshape(-1)
            chunks.append(torch.stack((1.0 - p1, p1), dim=1).cpu())
    return torch.cat(chunks).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="patient_*_pilot_p32_i2_s64_ma5_sa16_r0")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, region_records, case_records = {}, [], []

    for folder in sorted((HERE / "outputs").glob(args.pattern)):
        result_path, npz_path = folder / "result.json", folder / "dseg_lime_result.npz"
        if not result_path.exists() or not npz_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        fold = int(result["fold"])
        if fold not in models:
            checkpoint = best_checkpoint(REPO / "2DFold" / MODEL_FOLDERS[fold])
            model = RSNAAlternativeClassifierSingle2D(True, 1).to(device)
            model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
            model.eval()
            models[fold] = model
        model = models[fold]

        data = np.load(npz_path)
        image, segments = data["input_klf"].astype(np.float32), data["segments"]
        weights = data["weight_map"].astype(np.float32)
        explained_class = int(result["explained_class"])
        base_probability = float(data["class_probability"][explained_class])
        segment_ids = np.unique(segments)

        perturbed = np.repeat(image[None], len(segment_ids), axis=0)
        for index, segment_id in enumerate(segment_ids):
            perturbed[index][segments == segment_id] = 0.0
        perturbed_probabilities = class_probabilities(model, perturbed, device)

        local_weights, drops = [], []
        for index, segment_id in enumerate(segment_ids):
            mask = segments == segment_id
            weight = float(np.mean(weights[mask]))
            after = float(perturbed_probabilities[index, explained_class])
            drop = base_probability - after
            local_weights.append(weight)
            drops.append(drop)
            region_records.append({
                "patient": result["patient"], "fold": fold, "label": result["label"],
                "explained_class": explained_class, "segment_id": int(segment_id),
                "pixels": int(mask.sum()), "dseg_weight": weight,
                "base_probability": base_probability,
                "probability_after_deletion": after, "deletion_drop": drop,
            })

        local_weights, drops = np.asarray(local_weights), np.asarray(drops)
        top_weight_index, top_drop_index = int(np.argmax(local_weights)), int(np.argmax(drops))
        correlation = float(spearmanr(local_weights, drops).statistic) if len(segment_ids) > 1 else np.nan
        case_records.append({
            "patient": result["patient"], "fold": fold, "label": result["label"],
            "explained_class": explained_class, "segments": len(segment_ids),
            "sam_raw_masks": result.get("sam_raw_masks"), "base_probability": base_probability,
            "top_dseg_segment": int(segment_ids[top_weight_index]),
            "top_dseg_deletion_drop": float(drops[top_weight_index]),
            "max_observed_deletion_drop": float(drops[top_drop_index]),
            "top_region_is_most_faithful": top_weight_index == top_drop_index,
            "weight_drop_spearman": correlation,
        })

    if not case_records:
        raise SystemExit("Aucun résultat correspondant au motif demandé.")
    pilot_dir = HERE / "pilot"
    regions, cases = pd.DataFrame(region_records), pd.DataFrame(case_records)
    regions.to_csv(pilot_dir / "faithfulness_regions.csv", index=False)
    cases.to_csv(pilot_dir / "faithfulness_cases.csv", index=False)
    print(cases.to_string(index=False))
    print("\nDétails régionaux")
    print(regions.to_string(index=False))


if __name__ == "__main__":
    main()

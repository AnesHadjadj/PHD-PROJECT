"""Évaluation complémentaire du pilote DSEG-LIME sur 20 patients.

Produit des contrôles aléatoires à aire KLF égale, des courbes de suppression/
insertion et une comparaison spatiale explicite avec le masque KBF (KLF != 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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


def probabilities(model, images, explained_class, device):
    batch = torch.from_numpy(images[:, None].astype(np.float32)).to(device)
    output = []
    with torch.inference_mode():
        for local in batch.split(64):
            p1 = torch.sigmoid(model(local)).reshape(-1)
            output.append((p1 if explained_class == 1 else 1.0 - p1).cpu())
    return torch.cat(output).numpy()


def auc(x, y):
    order = np.argsort(x)
    return float(np.trapz(np.asarray(y)[order], np.asarray(x)[order]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="patient_*_pilot_p32_i2_s64_ma5_sa16_r0")
    parser.add_argument("--random-controls", type=int, default=100)
    parser.add_argument("--random-curves", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    models, cases, curves = {}, [], []

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
        image = data["input_klf"].astype(np.float32)
        segments = data["segments"].astype(np.int32)
        weights = data["weight_map"].astype(np.float32)
        explained_class = int(result["explained_class"])
        base = float(data["class_probability"][explained_class])
        ids = np.unique(segments)
        region_weights = np.array([weights[segments == sid].mean() for sid in ids])
        order = ids[np.argsort(region_weights)[::-1]]
        top_mask = segments == order[0]
        kbf_mask = image != 0
        effective_top = top_mask & kbf_mask
        kbf_count = int(kbf_mask.sum())

        top_deleted = image.copy()
        top_deleted[top_mask] = 0
        top_after = float(probabilities(model, top_deleted[None], explained_class, device)[0])
        top_drop = base - top_after

        # Contrôle négatif : mêmes voxels KLF supprimés, mais position aléatoire.
        active_indices = np.flatnonzero(kbf_mask)
        effective_count = int(effective_top.sum())
        random_images = np.repeat(image[None], args.random_controls, axis=0)
        for i in range(args.random_controls):
            chosen = rng.choice(active_indices, size=effective_count, replace=False)
            random_images[i].flat[chosen] = 0
        random_drops = base - probabilities(model, random_images, explained_class, device)
        p_random = float((1 + np.sum(random_drops >= top_drop)) / (args.random_controls + 1))

        # Courbes cumulatives DSEG : régions triées par poids décroissant.
        deletion_images, insertion_images = [], []
        deleted, inserted = image.copy(), np.zeros_like(image)
        covered = np.zeros_like(kbf_mask)
        fractions = [0.0]
        deletion_images.append(deleted.copy())
        insertion_images.append(inserted.copy())
        for sid in order:
            region = segments == sid
            deleted[region] = 0
            inserted[region] = image[region]
            covered |= region & kbf_mask
            fractions.append(float(covered.sum() / max(kbf_count, 1)))
            deletion_images.append(deleted.copy())
            insertion_images.append(inserted.copy())
        deletion_p = probabilities(model, np.stack(deletion_images), explained_class, device)
        insertion_p = probabilities(model, np.stack(insertion_images), explained_class, device)
        for step, (fraction, pd_, pi_) in enumerate(zip(fractions, deletion_p, insertion_p)):
            curves.append({
                "patient": result["patient"], "fold": fold, "curve": "dseg", "step": step,
                "fraction_klf": fraction, "deletion_probability": float(pd_),
                "insertion_probability": float(pi_),
            })

        # Référence de courbe : permutations aléatoires des mêmes régions.
        random_del_auc, random_ins_auc = [], []
        for _ in range(args.random_curves):
            random_order = rng.permutation(ids)
            deleted, inserted = image.copy(), np.zeros_like(image)
            covered = np.zeros_like(kbf_mask)
            fs, dimgs, iimgs = [0.0], [deleted.copy()], [inserted.copy()]
            for sid in random_order:
                region = segments == sid
                deleted[region] = 0
                inserted[region] = image[region]
                covered |= region & kbf_mask
                fs.append(float(covered.sum() / max(kbf_count, 1)))
                dimgs.append(deleted.copy())
                iimgs.append(inserted.copy())
            random_del_auc.append(auc(fs, probabilities(model, np.stack(dimgs), explained_class, device)))
            random_ins_auc.append(auc(fs, probabilities(model, np.stack(iimgs), explained_class, device)))

        positive = weights > 0
        inter = int((positive & kbf_mask).sum())
        union = int((positive | kbf_mask).sum())
        dice_denom = int(positive.sum() + kbf_mask.sum())
        dseg_del_auc, dseg_ins_auc = auc(fractions, deletion_p), auc(fractions, insertion_p)
        cases.append({
            "patient": result["patient"], "fold": fold, "label": result["label"],
            "explained_class": explained_class, "segments": len(ids), "base_probability": base,
            "top_deletion_drop": top_drop, "top_effective_klf_voxels": effective_count,
            "random_drop_mean": float(random_drops.mean()),
            "random_drop_q95": float(np.quantile(random_drops, 0.95)),
            "top_vs_random_p": p_random,
            "top_beats_random_mean": bool(top_drop > random_drops.mean()),
            "deletion_auc_dseg": dseg_del_auc,
            "deletion_auc_random_mean": float(np.mean(random_del_auc)),
            "insertion_auc_dseg": dseg_ins_auc,
            "insertion_auc_random_mean": float(np.mean(random_ins_auc)),
            "deletion_better_than_random": bool(dseg_del_auc < np.mean(random_del_auc)),
            "insertion_better_than_random": bool(dseg_ins_auc > np.mean(random_ins_auc)),
            "kbf_voxels": kbf_count, "positive_dseg_pixels": int(positive.sum()),
            "dseg_positive_kbf_precision": inter / max(int(positive.sum()), 1),
            "dseg_positive_kbf_recall": inter / max(kbf_count, 1),
            "dseg_positive_kbf_dice": 2 * inter / max(dice_denom, 1),
            "dseg_positive_kbf_iou": inter / max(union, 1),
            "top_region_kbf_fraction": effective_count / max(kbf_count, 1),
        })

    if not cases:
        raise SystemExit("Aucun résultat DSEG trouvé.")
    pilot = HERE / "pilot"
    case_df, curve_df = pd.DataFrame(cases), pd.DataFrame(curves)
    case_df.to_csv(pilot / "complete_evaluation_cases.csv", index=False)
    curve_df.to_csv(pilot / "progressive_curves.csv", index=False)
    summary = {
        "patients": len(case_df),
        "top_beats_random_mean_rate": float(case_df.top_beats_random_mean.mean()),
        "top_significant_vs_random_rate_p05": float((case_df.top_vs_random_p <= 0.05).mean()),
        "median_top_vs_random_p": float(case_df.top_vs_random_p.median()),
        "deletion_better_than_random_rate": float(case_df.deletion_better_than_random.mean()),
        "insertion_better_than_random_rate": float(case_df.insertion_better_than_random.mean()),
        "median_deletion_auc_gain": float((case_df.deletion_auc_random_mean - case_df.deletion_auc_dseg).median()),
        "median_insertion_auc_gain": float((case_df.insertion_auc_dseg - case_df.insertion_auc_random_mean).median()),
        "median_kbf_recall_positive_dseg": float(case_df.dseg_positive_kbf_recall.median()),
        "median_kbf_precision_positive_dseg": float(case_df.dseg_positive_kbf_precision.median()),
        "median_kbf_dice_positive_dseg": float(case_df.dseg_positive_kbf_dice.median()),
    }
    (pilot / "complete_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(case_df.top_deletion_drop, bins=10, color="#3366a6")
    axes[0, 0].set(title="Effet de suppression de la région DSEG principale", xlabel="Baisse de probabilité", ylabel="Patients")
    axes[0, 1].scatter(case_df.random_drop_mean, case_df.top_deletion_drop, c="#b43c39")
    lim = [min(0, case_df.random_drop_mean.min(), case_df.top_deletion_drop.min()), max(case_df.random_drop_mean.max(), case_df.top_deletion_drop.max())]
    axes[0, 1].plot(lim, lim, "k--", lw=1)
    axes[0, 1].set(title="DSEG contre contrôle aléatoire à aire égale", xlabel="Baisse aléatoire moyenne", ylabel="Baisse DSEG")
    axes[1, 0].scatter(case_df.deletion_auc_random_mean, case_df.deletion_auc_dseg, c="#4b8a5f")
    lim2 = [min(case_df.deletion_auc_random_mean.min(), case_df.deletion_auc_dseg.min()), max(case_df.deletion_auc_random_mean.max(), case_df.deletion_auc_dseg.max())]
    axes[1, 0].plot(lim2, lim2, "k--", lw=1)
    axes[1, 0].set(title="AUC de suppression (plus faible = mieux)", xlabel="Ordre aléatoire", ylabel="Ordre DSEG")
    axes[1, 1].hist(case_df.dseg_positive_kbf_recall, bins=10, color="#7b5aa6")
    axes[1, 1].set(title="Couverture du masque KBF par DSEG positif", xlabel="Rappel spatial", ylabel="Patients")
    fig.tight_layout()
    fig.savefig(pilot / "pilot_complete_summary.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

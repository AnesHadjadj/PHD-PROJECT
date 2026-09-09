"""First DSEG-LIME probe for the Capuozzo 2D KLF classifier."""

from __future__ import annotations

import argparse
import json
import sys
import hashlib
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

from baseline_2d_probe import (
    RSNAAlternativeClassifierSingle2D,
    best_checkpoint,
    load_klf,
    load_state_dict,
    most_informative_slice,
    patient_for_fold,
)


class RecordingMaskGenerator:
    """Décorateur léger qui enregistre les masques bruts produits par SAM."""

    def __init__(self, generator):
        self.generator = generator
        self.last_masks = []

    def generate(self, image):
        self.last_masks = self.generator.generate(image)
        return self.last_masks


def main():
    started = time.perf_counter()
    # ------------------------------------------------------------------
    # 1) Paramètres reproductibles de l'expérience
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--fixed-class', type=int, choices=[0, 1])
    parser.add_argument('--exact-mask', action='store_true')
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--min-segment-area", type=int, default=20)
    parser.add_argument("--sam-min-mask-area", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patient", type=Path,
                        help="Chemin facultatif vers un fichier *_KLF.mat précis")
    parser.add_argument("--output-suffix", default="",
                        help="Suffixe facultatif pour séparer les répétitions/configurations")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(r"D:\PHD\rsna-miccai-brats-dataset\raw\train_mat\KLF"),
    )
    parser.add_argument(
        "--folds-csv", type=Path,
        default=Path(r"D:\PHD\rsna-miccai-brats-dataset\raw\train_fold.csv"),
    )
    args = parser.parse_args()

    # Tous les chemins sont reconstruits depuis l'emplacement du script.
    # Le notebook peut donc être lancé sans dépendre du dossier courant.
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    dseg_root = workspace / "DSEG-LIME-main"
    sam_root = workspace / "segment-anything-main"
    sys.path.insert(0, str(sam_root))
    sys.path.insert(0, str(dseg_root))

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from Utilities.dseg_lime import DSEG_Lime

    model_folders = [
        "fold0_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_09-40-29",
        "fold1_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_10-15-21",
        "fold2_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_10-50-28",
        "fold3_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_09-42-53",
        "fold4_RSNAAlternativeClassifierSingle2D-DW-SO-KLF-m_Aug19_10-18-34",
    ]
    # ------------------------------------------------------------------
    # 2) Même patient, même fold et même coupe que la baseline Capuozzo
    # ------------------------------------------------------------------
    checkpoint = args.checkpoint or best_checkpoint(repo / "2DFold" / model_folders[args.fold])
    patient = args.patient or patient_for_fold(args.data_root, args.folds_csv, args.fold)
    volume, label = load_klf(patient)
    klf_image, slice_index, nonzero = most_informative_slice(volume)

    # ------------------------------------------------------------------
    # 3) Correspondance spatiale FLAIR / KLF
    # ------------------------------------------------------------------
    # SAM a besoin de texture anatomique pour former des régions cohérentes.
    # Le CNN, lui, doit impérativement continuer à recevoir KLF.
    # On prend donc la coupe FLAIR du même patient et au même indice.
    flair_path = (
        args.data_root.parent / "FLAIR" / patient.parent.name /
        patient.name.replace("_KLF.mat", "_FLAIR.mat")
    )
    flair_volume = loadmat(flair_path)["X"].astype(np.float32)
    if flair_volume.shape[0] > 192:
        flair_volume = flair_volume[32:224, 32:224, 32:224]
    fmin, fmax = float(flair_volume.min()), float(flair_volume.max())
    if fmax > fmin:
        flair_volume = (flair_volume - fmin) / (fmax - fmin)
    flair_image = flair_volume[:, :, slice_index]
    rgb = np.repeat(
        (np.clip(flair_image, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2
    )

    # ------------------------------------------------------------------
    # 4) Chargement du CNN original, sans réentraînement
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = RSNAAlternativeClassifierSingle2D(True, 1).to(device)
    classifier.load_state_dict(load_state_dict(checkpoint, device), strict=True)
    classifier.eval()

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        """Adapte les perturbations DSEG-LIME à l'entrée KLF du CNN.

        DSEG-LIME perturbe l'image RGB/FLAIR. Cette fonction récupère le
        masque des pixels conservés, l'applique à KLF, puis demande au CNN
        les probabilités des classes 0 et 1.
        """
        arr = np.asarray(images, dtype=np.float32)
        # Les pixels identiques à l'image originale sont les pixels conservés.
        active = np.all(arr == rgb[None, ...], axis=-1).astype(np.float32)
        # Le même masque spatial est transféré à l'image KLF originale.
        perturbed_klf = active * klf_image[None, :, :]
        # Forme attendue par PyTorch : [nombre d'images, canal, hauteur, largeur].
        batch = torch.from_numpy(perturbed_klf[:, None, :, :]).to(device)
        outputs = []
        with torch.inference_mode():
            for local_batch in batch.split(16):
                # Le modèle retourne un logit pour la classe MGMT positive.
                p1 = torch.sigmoid(classifier(local_batch)).reshape(-1)
                # LIME attend une probabilité pour chaque classe : P(0), P(1).
                outputs.append(torch.stack((1.0 - p1, p1), dim=1).cpu())
        return torch.cat(outputs).numpy()

    # ------------------------------------------------------------------
    # 5) Segmentation SAM et explication hiérarchique DSEG-LIME
    # ------------------------------------------------------------------
    sam_checkpoint = dseg_root / "Models" / "pretrained" / "sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint)).to(device)
    sam_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=32,
        min_mask_region_area=args.sam_min_mask_area,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.75,
    )
    # DSEG-LIME utilise l'interface generate(). Le décorateur permet de
    # conserver le nombre et la taille des propositions SAM sans modifier SAM.
    generator = RecordingMaskGenerator(sam_generator)

    # Tous les résultats d'un patient sont regroupés dans un seul dossier.
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_dir = repo / "contribution" / "outputs" / f"patient_{patient.stem[:5]}_fold{args.fold}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_png = output_dir / "klf_slice.png"
    plt.imsave(input_png, rgb)

    explainer = DSEG_Lime(random_state=args.seed, min_segment_area=args.min_segment_area)
    explanation = explainer.explain_instance(
        image=rgb.astype(np.float32) if args.exact_mask else rgb,
        classifier_fn=classifier_fn,
        feature_extractor=generator,
        model=sam,
        image_path=str(input_png),
        top_labels=2,
        num_features=1000,
        num_samples=args.num_samples,
        batch_size=16,
        iterations=args.iterations,
        random_seed=args.seed,
        progress_bar=True,
        hierarchy_label=args.fixed_class,
        hide_color=-1 if args.exact_mask else None,
    )

    # ------------------------------------------------------------------
    # 6) Conversion de l'explication en masques et résultats lisibles
    # ------------------------------------------------------------------
    predicted_label = int(explanation.top_labels[0]) if args.fixed_class is None else args.fixed_class
    _, positive_mask = explanation.get_image_and_mask(
        predicted_label, positive_only=True, num_features=5, hide_rest=True
    )
    weights = np.zeros_like(positive_mask, dtype=np.float32)
    for segment_id, weight, *_ in explanation.local_exp[predicted_label]:
        weights[explanation.segments == segment_id] = float(weight)

    # Probabilités du patient non perturbé, utilisées comme référence.
    probability = classifier_fn(rgb[None])[0]
    np.savez_compressed(
        output_dir / "dseg_lime_result.npz",
        input_klf=klf_image,
        input_flair=flair_image,
        rgb=rgb,
        segments=explanation.segments,
        positive_mask=positive_mask,
        weight_map=weights,
        class_probability=probability,
    )
    plt.imsave(output_dir / "segments.png", explanation.segments, cmap="nipy_spectral")
    plt.imsave(output_dir / "positive_mask.png", positive_mask, cmap="gray")
    plt.imsave(output_dir / "weight_map.png", weights, cmap="coolwarm")

    result = {
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'fixed_class': args.fixed_class,
        'exact_mask': args.exact_mask,
        'duration_seconds': time.perf_counter() - started,
        'surrogate_scores': {str(k): float(v) for k, v in explanation.score.items()},
        "patient": patient.stem.replace("_KLF", ""),
        "fold": args.fold,
        "label": label,
        "slice_index": slice_index,
        "slice_nonzero_voxels": nonzero,
        "class_probabilities": probability.tolist(),
        "explained_class": predicted_label,
        "segments": int(np.unique(explanation.segments).size),
        "segment_sizes": {
            str(int(segment_id)): int((explanation.segments == segment_id).sum())
            for segment_id in np.unique(explanation.segments)
        },
        "sam_raw_masks": len(generator.last_masks),
        "sam_raw_mask_areas": sorted(
            [int(mask.get("area", np.asarray(mask["segmentation"]).sum()))
             for mask in generator.last_masks],
            reverse=True,
        ),
        "num_samples_per_iteration": args.num_samples,
        "iterations": args.iterations,
        "points_per_side": args.points_per_side,
        "min_segment_area": args.min_segment_area,
        "sam_min_mask_area": args.sam_min_mask_area,
        "seed": args.seed,
        "output_dir": str(output_dir),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

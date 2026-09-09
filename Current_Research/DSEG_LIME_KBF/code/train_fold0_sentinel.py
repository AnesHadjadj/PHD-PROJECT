"""Entraîne le fold sentinelle 0 : KBF original contre KBF-DSEG.

Les 117 patients du fold 0 restent exclusivement réservés au test externe.
Les 468 autres patients sont séparés en apprentissage et validation interne.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_2d_probe import RSNAAlternativeClassifierSingle2D


HERE = Path(__file__).resolve().parent
FOLDS = Path(r"D:\PHD\rsna-miccai-brats-dataset\raw\train_fold.csv")
SUFFIX = "sentinel_fold0_p32_i2_s128_ma5_sa16_r0"
WORK = HERE / "sentinel_fold0"
STATUS = WORK / "status.json"
ALPHAS = (0.25, 0.50, 1.00)
SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15
BATCH = 32


def update_status(**values):
    state = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    state.update(values)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATUS.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def make_variant(image, weight, explained_class, alpha):
    if alpha is None:
        return image
    scale = max(float(np.max(np.abs(weight))), 1e-8)
    signed = weight / scale
    direction = 1.0 if explained_class == 1 else -1.0
    return image * np.clip(1.0 + alpha * direction * signed, 0.0, 2.0)


def metrics(y, probability):
    pred = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def predict(model, loader, device):
    probs, labels = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            probs.append(torch.sigmoid(model(x.to(device))).reshape(-1).cpu())
            labels.append(y.reshape(-1).cpu())
    return torch.cat(labels).numpy().astype(int), torch.cat(probs).numpy()


def train_one(name, images, labels, train_idx, inner_idx, outer_idx, device, experiment_start, variant_number):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = RSNAAlternativeClassifierSingle2D(True, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    tensor_x = torch.from_numpy(images[:, None].astype(np.float32))
    tensor_y = torch.from_numpy(labels.astype(np.float32))
    train_loader = DataLoader(TensorDataset(tensor_x[train_idx], tensor_y[train_idx]), batch_size=BATCH, shuffle=True)
    inner_loader = DataLoader(TensorDataset(tensor_x[inner_idx], tensor_y[inner_idx]), batch_size=BATCH)
    outer_loader = DataLoader(TensorDataset(tensor_x[outer_idx], tensor_y[outer_idx]), batch_size=BATCH)

    best_auc, best_epoch, no_improvement, best_state = -np.inf, 0, 0, None
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)).reshape(-1), y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        y_inner, p_inner = predict(model, inner_loader, device)
        inner_auc = roc_auc_score(y_inner, p_inner)
        history.append({"variant": name, "epoch": epoch, "train_loss": np.mean(losses), "inner_auc": inner_auc})
        if inner_auc > best_auc + 1e-4:
            best_auc, best_epoch, no_improvement = inner_auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
        elapsed = (time.time() - experiment_start) / 60
        update_status(stage="training", state="running", variant=name, variant_number=variant_number,
                      variants_total=4, epoch=epoch, max_epochs=MAX_EPOCHS,
                      best_inner_auc=round(float(best_auc), 5), elapsed_minutes=round(elapsed, 2))
        if no_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    y_outer, p_outer = predict(model, outer_loader, device)
    torch.save(best_state, WORK / f"{name}_best.pth")
    return {
        "variant": name, "best_epoch": best_epoch, "epochs_run": epoch,
        "inner_auc": float(best_auc), **metrics(y_outer, p_outer),
    }, history, y_outer, p_outer


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
experiment_start = time.time()
update_status(stage="loading", state="running", completed=0, total=585,
              training_started_at=datetime.now().isoformat(timespec="seconds"))

folds = pd.read_csv(FOLDS)
images, weights, explained, labels, patient_ids = [], [], [], [], []
for position, row in enumerate(folds.itertuples(index=False), 1):
    patient = str(int(row.BraTS21ID)).zfill(5)
    folder = HERE / "outputs" / f"patient_{patient}_fold0_{SUFFIX}"
    data = np.load(folder / "dseg_lime_result.npz")
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    images.append(data["input_klf"].astype(np.float32))
    weights.append(data["weight_map"].astype(np.float32))
    explained.append(int(result["explained_class"]))
    labels.append(int(row.MGMT_value))
    patient_ids.append(patient)
    if position % 50 == 0 or position == len(folds):
        update_status(completed=position, percent=round(100 * position / len(folds), 2))

images, weights = np.stack(images), np.stack(weights)
explained, labels = np.asarray(explained), np.asarray(labels)
outer_idx = np.flatnonzero(folds.fold.to_numpy() == 0)
development_idx = np.flatnonzero(folds.fold.to_numpy() != 0)
train_idx, inner_idx = train_test_split(
    development_idx, test_size=0.15, random_state=SEED, stratify=labels[development_idx]
)

variants = [("original", None)] + [(f"signed_a{alpha:.2f}", alpha) for alpha in ALPHAS]
all_results, all_history, prediction_rows = [], [], []
for number, (name, alpha) in enumerate(variants, 1):
    transformed = np.stack([
        make_variant(image, weight, klass, alpha)
        for image, weight, klass in zip(images, weights, explained)
    ])
    result, history, y_outer, p_outer = train_one(
        name, transformed, labels, train_idx, inner_idx, outer_idx, device, experiment_start, number
    )
    all_results.append(result)
    all_history.extend(history)
    for idx, y, p in zip(outer_idx, y_outer, p_outer):
        prediction_rows.append({"patient": patient_ids[idx], "label": int(y), "variant": name,
                                "p_mgmt_positive": float(p), "prediction": int(p >= 0.5)})

pd.DataFrame(all_results).to_csv(WORK / "fold0_metrics.csv", index=False)
pd.DataFrame(all_history).to_csv(WORK / "training_history.csv", index=False)
pd.DataFrame(prediction_rows).to_csv(WORK / "fold0_predictions.csv", index=False)
update_status(stage="complete", state="complete", elapsed_minutes=round((time.time() - experiment_start) / 60, 2),
              metrics_file=str(WORK / "fold0_metrics.csv"), estimated_remaining_minutes=0)
print(pd.DataFrame(all_results).to_string(index=False))

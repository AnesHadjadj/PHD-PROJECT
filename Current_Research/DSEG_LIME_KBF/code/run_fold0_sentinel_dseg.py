"""Génère les explications DSEG nécessaires au fold sentinelle 0.

Le checkpoint du fold 0 (entraîné sans les 117 patients du fold 0) est utilisé
pour tous les patients. Un état JSON est actualisé après chaque patient.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = Path(r"D:\PHD\rsna-miccai-brats-dataset\raw\train_mat\KLF")
FOLDS = Path(r"D:\PHD\rsna-miccai-brats-dataset\raw\train_fold.csv")
PROBE = HERE / "dseg_lime_mri_probe.py"
STATUS = HERE / "sentinel_fold0" / "status.json"
SUFFIX = "sentinel_fold0_p32_i2_s128_ma5_sa16_r0"


def write_status(**values):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if STATUS.exists():
        current = json.loads(STATUS.read_text(encoding="utf-8"))
    current.update(values)
    current["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATUS.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


folds = pd.read_csv(FOLDS)
total = len(folds)
started = time.time()
done = 0
write_status(stage="dseg_generation", state="running", total=total, completed=0,
             started_at=datetime.now().isoformat(timespec="seconds"), gpu="RTX 3060 Laptop 6 GB")

try:
    for row in folds.itertuples(index=False):
        patient = str(int(row.BraTS21ID)).zfill(5)
        label = str(int(row.MGMT_value))
        source = DATA / label / f"{patient}_KLF.mat"
        result = HERE / "outputs" / f"patient_{patient}_fold0_{SUFFIX}" / "result.json"
        if not result.exists():
            command = [
                sys.executable, str(PROBE), "--fold", "0", "--patient", str(source),
                "--num-samples", "128", "--points-per-side", "32", "--iterations", "2",
                "--min-segment-area", "5", "--sam-min-mask-area", "16", "--seed", "42",
                "--output-suffix", SUFFIX,
            ]
            subprocess.run(command, check=True)
        done += 1
        elapsed = time.time() - started
        rate = elapsed / done
        write_status(
            completed=done, percent=round(100 * done / total, 2),
            last_patient=patient, elapsed_minutes=round(elapsed / 60, 2),
            estimated_remaining_minutes=round(rate * (total - done) / 60, 2),
        )
    write_status(state="complete", completed=total, percent=100.0,
                 elapsed_minutes=round((time.time() - started) / 60, 2), estimated_remaining_minutes=0)
except Exception as exc:
    write_status(state="failed", error=repr(exc), completed=done)
    raise

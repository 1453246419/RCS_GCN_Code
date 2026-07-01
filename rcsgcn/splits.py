from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_splits(
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(labels)), labels)
    else:
        groups = np.asarray(groups).reshape(-1)
        if len(groups) != len(labels):
            raise ValueError("Group and label counts do not match.")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(labels)), labels, groups)
    return [(train.astype(np.int64), test.astype(np.int64)) for train, test in iterator]


def save_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    output_dir: str | Path,
    seed: int,
    grouped: bool,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    manifest: dict[str, object] = {
        "seed": seed,
        "n_splits": len(splits),
        "protocol": "patient-grouped stratified" if grouped else "sample-level stratified",
        "folds": [],
    }
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        np.savez_compressed(output_dir / f"fold_{fold}.npz", train_idx=train_idx, test_idx=test_idx)
        manifest["folds"].append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_counts": dict(sorted(Counter(labels[train_idx].tolist()).items())),
            "test_counts": dict(sorted(Counter(labels[test_idx].tolist()).items())),
        })
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_fold(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as split:
        return split["train_idx"].astype(np.int64), split["test_idx"].astype(np.int64)


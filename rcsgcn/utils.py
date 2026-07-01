from __future__ import annotations

import csv
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without touching private data."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def load_label_bundle(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load labels from the original pickle format, NPY, JSON, or CSV."""
    path = Path(path)
    suffix = path.suffix.lower()
    names: list[str] | None = None

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            obj = pickle.load(handle, encoding="latin1")
        if isinstance(obj, tuple) and len(obj) >= 2:
            names = [str(item) for item in obj[0]]
            labels = np.asarray(obj[1], dtype=np.int64)
        else:
            labels = np.asarray(obj, dtype=np.int64)
    elif suffix == ".npy":
        labels = np.asarray(np.load(path), dtype=np.int64)
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            labels = np.asarray(obj["labels"], dtype=np.int64)
            names = [str(item) for item in obj.get("sample_names", [])] or None
        else:
            labels = np.asarray(obj, dtype=np.int64)
    elif suffix == ".csv":
        labels_list: list[int] = []
        names_list: list[str] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV labels require a 'label' column.")
            for row_index, row in enumerate(reader):
                labels_list.append(int(row["label"]))
                names_list.append(row.get("sample_name") or str(row_index))
        labels = np.asarray(labels_list, dtype=np.int64)
        names = names_list
    else:
        raise ValueError(f"Unsupported label format: {path.suffix}")

    labels = labels.reshape(-1)
    if names is None:
        names = [str(index) for index in range(len(labels))]
    if len(names) != len(labels):
        raise ValueError("Sample-name and label counts do not match.")
    return names, labels


def load_vector(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path)).reshape(-1)
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8"))).reshape(-1)
    with path.open("r", encoding="utf-8-sig") as handle:
        return np.asarray([line.strip() for line in handle if line.strip()])


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    return {str(key).removeprefix("module."): value for key, value in checkpoint.items()}


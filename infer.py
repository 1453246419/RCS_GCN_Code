from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rcsgcn.dataset import SkeletonDataset
from rcsgcn.metrics import classification_metrics
from rcsgcn.model import RCSGCN
from rcsgcn.preprocessing import build_12_channel, validate_12_channel, validate_joint_data
from rcsgcn.splits import load_fold
from rcsgcn.utils import load_label_bundle, load_yaml, resolve_device, set_seed, unwrap_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete RCS-GCN checkpoint inference.")
    parser.add_argument("--config", default="configs/rcs_gcn_12ch.yaml")
    parser.add_argument("--data", required=True, help="Raw 3-channel or prepared 12-channel NPY.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint created by train_cv.py.")
    parser.add_argument("--labels", help="Optional private labels for metric calculation.")
    parser.add_argument("--split", help="Optional fold_N.npz; inference uses its test_idx.")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--whole-graph", action="store_true", help="Disable informative masks for an ablation check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(args.device or config["device"])
    data = np.load(args.data, mmap_mode="r")
    if data.ndim != 5:
        raise ValueError(f"Expected a five-dimensional NPY, got {data.shape}.")
    if data.shape[1] == 3:
        validate_joint_data(data)
        print("[INFO] Building 12-channel input in memory from raw joint data.")
        data = build_12_channel(np.asarray(data))
    else:
        validate_12_channel(data)

    if args.labels:
        sample_names, labels = load_label_bundle(args.labels)
        if len(labels) != len(data):
            raise ValueError("Data and label counts do not match.")
        has_labels = True
    else:
        sample_names = [str(index) for index in range(len(data))]
        labels = np.zeros(len(data), dtype=np.int64)
        has_labels = False
    indices = load_fold(args.split)[1] if args.split else np.arange(len(data), dtype=np.int64)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", config["model"]) if isinstance(checkpoint, dict) else config["model"]
    model = RCSGCN(**model_config)
    state = unwrap_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    dataset = SkeletonDataset(data, labels, indices)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"] if args.num_workers is None else args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true: list[int] = []
    y_pred: list[int] = []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["sample_index", "sample_name", "y_true", "y_pred", "p0", "p1", "p2", "p3"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for batch, truth, batch_indices in loader:
                batch = batch.to(device, non_blocking=True)
                logits = model(batch, use_informative_mask=not args.whole_graph)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = probabilities.argmax(axis=1)
                for index, label, prediction, probability in zip(
                    batch_indices.tolist(), truth.tolist(), predictions.tolist(), probabilities
                ):
                    row = {
                        "sample_index": index,
                        "sample_name": sample_names[index],
                        "y_true": label if has_labels else "",
                        "y_pred": prediction,
                    }
                    row.update({f"p{class_index}": float(value) for class_index, value in enumerate(probability)})
                    writer.writerow(row)
                    if has_labels:
                        y_true.append(int(label))
                        y_pred.append(int(prediction))
    print(f"[DONE] Predictions: {output_path}")
    if has_labels:
        metrics = classification_metrics(y_true, y_pred)
        metrics_path = output_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

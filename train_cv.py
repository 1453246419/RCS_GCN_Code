from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:  # Progress bars are optional for audit-only environments.
    def tqdm(iterable, **_kwargs):
        return iterable

from rcsgcn.dataset import SkeletonDataset
from rcsgcn.metrics import classification_metrics
from rcsgcn.model import RCSGCN
from rcsgcn.preprocessing import validate_12_channel
from rcsgcn.splits import load_fold, make_splits, save_splits
from rcsgcn.utils import load_label_bundle, load_yaml, resolve_device, seed_worker, set_seed


def class_weights(labels: np.ndarray, num_class: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_class).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Every training fold must contain all classes; counts={counts.tolist()}.")
    weights = len(labels) / (num_class * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
    data: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        SkeletonDataset(data, labels, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def set_backbone_trainable(model: RCSGCN, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("informative_mask."):
            parameter.requires_grad_(trainable)


def set_mask_trainable(model: RCSGCN, trainable: bool) -> None:
    for parameter in model.informative_mask.parameters():
        parameter.requires_grad_(trainable)


def warmup_epoch(
    model: RCSGCN,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_norm: float,
) -> float:
    model.train()
    set_backbone_trainable(model, True)
    set_mask_trainable(model, False)
    losses: list[float] = []
    for data, labels, _ in tqdm(loader, desc="warmup", leave=False):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model.predict_whole(data), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def alternating_epoch(
    model: RCSGCN,
    loader: DataLoader,
    criterion: torch.nn.Module,
    backbone_optimizer: torch.optim.Optimizer,
    mask_optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_fp: float,
    lambda_inv: float,
    clip_norm: float,
) -> dict[str, float]:
    totals = {"backbone_loss": [], "mask_loss": [], "classification": [], "perturbation": [], "invariance": []}
    for data, labels, _ in tqdm(loader, desc="alternating", leave=False):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Step 1: update only the informative mask while the feature extractor is fixed.
        model.eval()
        model.informative_mask.train()
        set_backbone_trainable(model, False)
        set_mask_trainable(model, True)
        mask_optimizer.zero_grad(set_to_none=True)
        masks = model.informative_mask(sample=True)
        logits_inf = model.predict_informative(data, masks)
        logits_non = model.predict_noninformative(data, masks)
        logits_rec = model.predict_recombined(data, masks)
        loss_cls = criterion(logits_inf, labels)
        loss_fp = -criterion(logits_non, labels)
        loss_inv = F.mse_loss(torch.softmax(logits_inf, dim=1), torch.softmax(logits_rec, dim=1))
        loss_mask = loss_cls + lambda_fp * loss_fp + lambda_inv * loss_inv
        loss_mask.backward()
        torch.nn.utils.clip_grad_norm_(model.informative_mask.parameters(), clip_norm)
        mask_optimizer.step()

        # Step 2: update the dual-stream backbone with a detached sampled mask.
        model.train()
        set_backbone_trainable(model, True)
        set_mask_trainable(model, False)
        backbone_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            detached_masks = tuple(mask.detach() for mask in model.informative_mask(sample=True))
        logits_inf = model.predict_informative(data, detached_masks)
        logits_rec = model.predict_recombined(data, detached_masks)
        loss_cls_backbone = criterion(logits_inf, labels)
        loss_inv_backbone = F.mse_loss(
            torch.softmax(logits_inf, dim=1), torch.softmax(logits_rec, dim=1)
        )
        loss_backbone = loss_cls_backbone + lambda_inv * loss_inv_backbone
        loss_backbone.backward()
        backbone_parameters = [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith("informative_mask.")
        ]
        torch.nn.utils.clip_grad_norm_(backbone_parameters, clip_norm)
        backbone_optimizer.step()

        totals["backbone_loss"].append(float(loss_backbone.detach().cpu()))
        totals["mask_loss"].append(float(loss_mask.detach().cpu()))
        totals["classification"].append(float(loss_cls.detach().cpu()))
        totals["perturbation"].append(float(loss_fp.detach().cpu()))
        totals["invariance"].append(float(loss_inv.detach().cpu()))
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def evaluate(model: RCSGCN, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict[str, Any]] = []
    for data, labels, indices in tqdm(loader, desc="evaluate", leave=False):
        data = data.to(device, non_blocking=True)
        probabilities = torch.softmax(model(data, use_informative_mask=True), dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        for index, truth, prediction, probability in zip(indices.tolist(), labels.tolist(), predictions.tolist(), probabilities):
            y_true.append(int(truth))
            y_pred.append(int(prediction))
            row: dict[str, Any] = {"sample_index": int(index), "y_true": int(truth), "y_pred": int(prediction)}
            row.update({f"p{class_index}": float(value) for class_index, value in enumerate(probability)})
            rows.append(row)
    return classification_metrics(y_true, y_pred), rows


def write_predictions(path: Path, rows: list[dict[str, Any]], sample_names: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_index", "sample_name", "y_true", "y_pred", "p0", "p1", "p2", "p3"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["sample_name"] = sample_names[int(row["sample_index"])]
            writer.writerow(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RCS-GCN using deterministic stratified cross-validation.")
    parser.add_argument("--config", default="configs/rcs_gcn_12ch.yaml")
    parser.add_argument("--data")
    parser.add_argument("--labels")
    parser.add_argument("--split-dir")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--folds", type=int, nargs="+", help="One-based folds; defaults to all configured folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(args.device or config["device"])
    data_path = Path(args.data or config["data"]["data_path"])
    label_path = Path(args.labels or config["data"]["label_path"])
    split_dir = Path(args.split_dir or config["data"]["split_dir"])
    output_dir = Path(args.output or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path, mmap_mode="r")
    validate_12_channel(data)
    sample_names, labels = load_label_bundle(label_path)
    if len(data) != len(labels):
        raise ValueError(f"Data/label mismatch: {len(data)} versus {len(labels)}.")

    split_config = config["split"]
    n_splits = int(split_config["n_splits"])
    if not (split_dir / "fold_1.npz").exists():
        splits = make_splits(labels, n_splits=n_splits, seed=int(split_config["random_state"]))
        save_splits(splits, labels, split_dir, seed=int(split_config["random_state"]), grouped=False)
        print(f"[INFO] Created deterministic sample-level splits in {split_dir}")

    selected_folds = args.folds or list(range(1, n_splits + 1))
    training = config["training"]
    epochs = int(args.epochs or training["epochs"])
    warmup_epochs = int(training["warmup_epochs"] if args.warmup_epochs is None else args.warmup_epochs)
    batch_size = int(args.batch_size or training["batch_size"])
    num_workers = int(config["data"]["num_workers"] if args.num_workers is None else args.num_workers)
    model_config = config["model"]

    for fold in selected_folds:
        if fold < 1 or fold > n_splits:
            raise ValueError(f"Fold {fold} is outside 1..{n_splits}.")
        fold_seed = seed + fold
        set_seed(fold_seed)
        train_idx, test_idx = load_fold(split_dir / f"fold_{fold}.npz")
        print(f"[FOLD {fold}] train={Counter(labels[train_idx])} test={Counter(labels[test_idx])}")
        train_loader = make_loader(
            data, labels, train_idx, batch_size, num_workers,
            shuffle=True, seed=fold_seed,
        )
        test_loader = make_loader(
            data, labels, test_idx, int(training["eval_batch_size"]), num_workers,
            shuffle=False, seed=fold_seed,
        )
        model = RCSGCN(**model_config).to(device)
        criterion = torch.nn.CrossEntropyLoss(class_weights(labels[train_idx], int(model_config["num_class"]), device))
        backbone_parameters = [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith("informative_mask.")
        ]
        backbone_optimizer = torch.optim.SGD(
            backbone_parameters,
            lr=float(training["learning_rate"]),
            momentum=float(training["momentum"]),
            nesterov=bool(training["nesterov"]),
            weight_decay=float(training["weight_decay"]),
        )
        mask_optimizer = torch.optim.Adam(
            model.informative_mask.parameters(),
            lr=float(training["mask_learning_rate"]),
            weight_decay=float(training["mask_weight_decay"]),
        )
        backbone_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            backbone_optimizer, milestones=list(training["lr_milestones"]), gamma=float(training["lr_gamma"])
        )
        mask_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            mask_optimizer, milestones=list(training["lr_milestones"]), gamma=float(training["lr_gamma"])
        )
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            if epoch <= warmup_epochs:
                loss = warmup_epoch(
                    model, train_loader, criterion, backbone_optimizer, device,
                    float(training["gradient_clip_norm"]),
                )
                epoch_row: dict[str, Any] = {"epoch": epoch, "stage": "warmup", "train_loss": loss}
            else:
                epoch_row = {"epoch": epoch, "stage": "alternating"}
                epoch_row.update(alternating_epoch(
                    model, train_loader, criterion, backbone_optimizer, mask_optimizer, device,
                    float(training["lambda_feature_perturbation"]),
                    float(training["lambda_invariance"]),
                    float(training["gradient_clip_norm"]),
                ))
                mask_scheduler.step()
            backbone_scheduler.step()
            history.append(epoch_row)
            print(f"[FOLD {fold}] epoch={epoch}/{epochs} {epoch_row}")

        metrics, rows = evaluate(model, test_loader, device)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": model_config,
            "fold": fold,
            "seed": fold_seed,
            "split_seed": int(split_config["random_state"]),
            "epoch": epochs,
            "metrics": metrics,
        }
        torch.save(checkpoint, fold_dir / "last_model.pt")
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (fold_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        write_predictions(fold_dir / "predictions.csv", rows, sample_names)
        print(f"[FOLD {fold}] final={metrics}")
    print(f"[DONE] Results written to {output_dir}")


if __name__ == "__main__":
    main()

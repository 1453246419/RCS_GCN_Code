# RCS-GCN: Code-Only Research Release

This repository contains the executable data pipeline, deterministic split
generation, model definition, cross-validation training logic, and checkpoint
inference program for RCS-GCN. It intentionally contains **no clinical data,
sample identifiers, labels, trained weights, or paper-result files**.

The release is intended to make the implementation logic auditable while
respecting clinical privacy and institutional data-governance requirements. It
cannot reproduce the numerical results in the paper without an authorized
dataset and a checkpoint trained on that dataset.

## What is included

- Construction of a 12-channel tensor from raw skeletons.
- Fixed-seed sample-level stratified 5-fold split generation.
- Optional patient-grouped split generation when private patient IDs are
  available.
- The 25-node graph, dual-stream GCN, Consistent Informative Masking Module,
  and Inter-Feature Sharing Blocks.
- Warm-up followed by alternating mask/backbone optimization.
- Weighted cross-entropy with fold-specific training counts.
- Complete checkpoint inference with class probabilities and optional metrics.
- Synthetic unit tests that do not use clinical data.

The completed synthetic execution audit is documented in
[SMOKE_TEST_REPORT.md](SMOKE_TEST_REPORT.md).

## Data contract

Raw skeleton data must have shape:

```text
(N, 3, T, 25, 1)
```

The three channels are `(x, y, confidence)`. The preprocessing output is:

```text
(N, 12, T, 25, 1)
```

with this fixed channel order:

```text
0:3   joint
3:6   bone
6:9   joint motion
9:12  bone motion
```

The confidence channel is retained as the third coordinate of every stream.
No clinical file should be copied into this repository.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 1. Build the 12-channel input

Run this command from the repository root. Keep both input and output outside
the Git repository.

```bash
python -m scripts.prepare_12ch \
  --input /private/path/all_data.npy \
  --output /private/path/all_data_12ch.npy
```

On Windows:

```powershell
python -m scripts.prepare_12ch `
  --input D:\private\all_data.npy `
  --output D:\private\all_data_12ch.npy
```

The script writes a small `*.metadata.json` file containing shapes and channel
order, but no sample values.

## 2. Generate deterministic folds

The paper-facing default is sample-level stratified 5-fold cross-validation:

```bash
python -m scripts.make_splits \
  --labels /private/path/all_label.pkl \
  --output-dir /private/path/splits_seed42 \
  --n-splits 5 \
  --seed 42
```

If patient IDs become available, create grouped folds explicitly:

```bash
python -m scripts.make_splits \
  --labels /private/path/all_label.pkl \
  --groups /private/path/patient_ids.npy \
  --output-dir /private/path/patient_splits_seed42 \
  --n-splits 5 \
  --seed 42
```

Patient IDs are never inferred from filenames. Each output directory contains
`fold_1.npz` through `fold_5.npz` and a count-only `manifest.json`.

## 3. Sanity-check one fold

Use a short run before launching the full experiment:

```bash
python train_cv.py \
  --config configs/rcs_gcn_12ch.yaml \
  --data /private/path/all_data_12ch.npy \
  --labels /private/path/all_label.pkl \
  --split-dir /private/path/splits_seed42 \
  --output /private/path/debug_run \
  --device cuda:0 \
  --folds 1 \
  --epochs 1
```

To test the alternating stage immediately on non-clinical debug data, append
`--warmup-epochs 0`. This override is for software checks, not the reported
experiment.

Windows PowerShell:

```powershell
python train_cv.py `
  --config configs\rcs_gcn_12ch.yaml `
  --data D:\private\all_data_12ch.npy `
  --labels D:\private\all_label.pkl `
  --split-dir D:\private\splits_seed42 `
  --output D:\private\debug_run `
  --device cuda:0 `
  --folds 1 `
  --epochs 1
```

## 4. Run the five folds

```bash
python train_cv.py \
  --config configs/rcs_gcn_12ch.yaml \
  --data /private/path/all_data_12ch.npy \
  --labels /private/path/all_label.pkl \
  --split-dir /private/path/splits_seed42 \
  --output /private/path/rcs_gcn_results \
  --device cuda:0 \
  --folds 1 2 3 4 5
```

Each fold writes the following private artifacts:

```text
fold_N/
  last_model.pt
  history.json
  metrics.json
  predictions.csv
```

These paths are ignored by the repository and must not be committed.

## 5. Checkpoint inference

Inference accepts raw 3-channel data or prepared 12-channel data. The latter is
recommended for speed and exact auditing of the channel order.

```bash
python infer.py \
  --config configs/rcs_gcn_12ch.yaml \
  --data /private/path/all_data_12ch.npy \
  --checkpoint /private/path/rcs_gcn_results/fold_1/last_model.pt \
  --split /private/path/splits_seed42/fold_1.npz \
  --labels /private/path/all_label.pkl \
  --output /private/path/fold_1_predictions.csv \
  --device cuda:0
```

Without `--labels`, the program still writes predictions and probabilities but
does not calculate metrics. `--whole-graph` runs the no-mask ablation path.

## Tests

The tests use synthetic arrays only:

```bash
python -m pytest -q
```

They verify channel construction, split reproducibility, disjoint folds, model
output shape, and finite logits.

## Reproducibility notes

- Global split seed: `42`.
- Model initialization seed for fold `k`: `42 + k`.
- Split protocol: `StratifiedKFold(n_splits=5, shuffle=True,
  random_state=42)` unless private group IDs are explicitly supplied.
- Class weights are `N / (K * n_c)` and are computed from the training indices
  of each fold only.
- The first 20 epochs use whole-graph warm-up. Remaining epochs alternate
  informative-mask and backbone updates.
- A fixed final epoch is evaluated. The held-out fold is not used for early
  stopping or checkpoint selection.

## Privacy and scope

See [DATA_AND_MODEL_POLICY.md](DATA_AND_MODEL_POLICY.md). The public README and
configuration use placeholder paths. Before publication, run the release
checklist to confirm that no private path, identifier, data array, prediction,
or checkpoint is tracked by Git.

## Upstream attribution

The graph-convolution backbone follows the 2s-AGCN implementation lineage.
See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) and retain the upstream
citation and license notice when publishing this repository.

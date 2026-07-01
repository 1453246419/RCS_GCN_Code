from __future__ import annotations

import argparse

from rcsgcn.splits import make_splits, save_splits
from rcsgcn.utils import load_label_bundle, load_vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic stratified cross-validation indices.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", help="Optional private patient/group IDs. No IDs are inferred from filenames.")
    args = parser.parse_args()

    _, labels = load_label_bundle(args.labels)
    groups = load_vector(args.groups) if args.groups else None
    splits = make_splits(labels, n_splits=args.n_splits, seed=args.seed, groups=groups)
    save_splits(splits, labels, args.output_dir, seed=args.seed, grouped=groups is not None)
    print(f"[DONE] Saved {len(splits)} folds to {args.output_dir}")


if __name__ == "__main__":
    main()


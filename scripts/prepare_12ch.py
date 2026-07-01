from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from rcsgcn.preprocessing import build_12_channel, validate_joint_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 12-channel RCS-GCN input without copying private data.")
    parser.add_argument("--input", required=True, help="Private raw NPY with shape (N,3,T,25,1).")
    parser.add_argument("--output", required=True, help="Destination NPY with shape (N,12,T,25,1).")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")
    joint = np.load(input_path, mmap_mode="r")
    validate_joint_data(joint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = open_memmap(
        output_path, mode="w+", dtype="float32",
        shape=(joint.shape[0], 12, joint.shape[2], 25, 1),
    )
    for start in range(0, len(joint), args.chunk_size):
        end = min(start + args.chunk_size, len(joint))
        output[start:end] = build_12_channel(np.asarray(joint[start:end]))
        print(f"[PREPARE] {end}/{len(joint)}")
    output.flush()
    metadata = {
        "source_shape": list(joint.shape),
        "output_shape": list(output.shape),
        "channel_order": ["joint", "bone", "joint_motion", "bone_motion"],
        "coordinate_order": ["x", "y", "confidence"],
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] {output_path} {tuple(output.shape)}")


if __name__ == "__main__":
    main()


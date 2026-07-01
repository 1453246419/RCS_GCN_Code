from __future__ import annotations

import numpy as np


# The 25-node topology used by the supplied MG skeleton data.
BONE_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 0), (2, 1), (3, 2), (4, 3),
    (5, 1), (6, 5), (7, 6),
    (8, 12), (9, 8), (10, 9), (11, 10),
    (13, 12), (15, 0), (16, 0), (17, 15),
    (19, 20), (21, 14), (22, 11), (23, 22),
    (24, 11), (14, 13), (8, 1), (16, 18), (19, 14),
)


def validate_joint_data(data: np.ndarray) -> None:
    if data.ndim != 5:
        raise ValueError(f"Expected (N,C,T,V,M), got {data.shape}.")
    if data.shape[1] != 3:
        raise ValueError(f"Raw joint input requires 3 channels (x,y,confidence), got {data.shape[1]}.")
    if data.shape[3] != 25:
        raise ValueError(f"This release expects 25 joints, got {data.shape[3]}.")
    if data.shape[4] != 1:
        raise ValueError(f"This MG setting expects one person (M=1), got M={data.shape[4]}.")


def validate_12_channel(data: np.ndarray) -> None:
    if data.ndim != 5 or data.shape[1] != 12 or data.shape[3:] != (25, 1):
        raise ValueError(f"Expected (N,12,T,25,1), got {data.shape}.")


def compute_bone(joint: np.ndarray) -> np.ndarray:
    validate_joint_data(joint)
    bone = np.zeros_like(joint, dtype=np.float32)
    for child, parent in BONE_PAIRS:
        bone[:, :, :, child, :] = joint[:, :, :, child, :] - joint[:, :, :, parent, :]
    return bone


def compute_motion(stream: np.ndarray) -> np.ndarray:
    if stream.ndim != 5:
        raise ValueError(f"Expected a five-dimensional stream, got {stream.shape}.")
    motion = np.zeros_like(stream, dtype=np.float32)
    motion[:, :, :-1, :, :] = stream[:, :, 1:, :, :] - stream[:, :, :-1, :, :]
    return motion


def build_12_channel(joint: np.ndarray, replace_nonfinite: bool = True) -> np.ndarray:
    """Create [joint, bone, joint motion, bone motion] in this exact order."""
    joint = np.asarray(joint, dtype=np.float32)
    validate_joint_data(joint)
    if replace_nonfinite:
        joint = np.nan_to_num(joint, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    elif not np.isfinite(joint).all():
        raise ValueError("Raw joint data contains NaN or Inf values.")
    bone = compute_bone(joint)
    joint_motion = compute_motion(joint)
    bone_motion = compute_motion(bone)
    output = np.concatenate((joint, bone, joint_motion, bone_motion), axis=1)
    validate_12_channel(output)
    return output.astype(np.float32, copy=False)


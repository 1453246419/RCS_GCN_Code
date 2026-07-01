from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import validate_12_channel


class SkeletonDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        validate_12_channel(data)
        self.data = data
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        if len(self.data) != len(self.labels):
            raise ValueError("Data and label counts do not match.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        sample = np.asarray(self.data[index], dtype=np.float32).copy()
        return torch.from_numpy(sample), torch.tensor(self.labels[index], dtype=torch.long), index

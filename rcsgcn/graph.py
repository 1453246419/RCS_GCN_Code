from __future__ import annotations

import numpy as np

from .preprocessing import BONE_PAIRS


def normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=0)
    inverse = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inverse[nonzero] = 1.0 / degree[nonzero]
    return adjacency @ np.diag(inverse)


def edge_to_matrix(edges: list[tuple[int, int]], num_node: int) -> np.ndarray:
    matrix = np.zeros((num_node, num_node), dtype=np.float32)
    for source, target in edges:
        matrix[target, source] = 1.0
    return matrix


class Graph:
    """Three-subset spatial graph for the 25-node MG skeleton."""

    def __init__(self, num_node: int = 25):
        if num_node != 25:
            raise ValueError("The released topology is defined for 25 joints.")
        self.num_node = num_node
        self.self_links = [(index, index) for index in range(num_node)]
        self.inward = list(BONE_PAIRS)
        self.outward = [(target, source) for source, target in self.inward]
        identity = edge_to_matrix(self.self_links, num_node)
        inward = normalize_digraph(edge_to_matrix(self.inward, num_node))
        outward = normalize_digraph(edge_to_matrix(self.outward, num_node))
        self.A = np.stack((identity, inward, outward)).astype(np.float32)


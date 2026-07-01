from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import Graph


def _conv_init(layer: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _bn_init(layer: nn.BatchNorm2d, scale: float = 1.0) -> None:
    nn.init.constant_(layer.weight, scale)
    nn.init.zeros_(layer.bias)


class InformativeMask(nn.Module):
    """Shared node and edge masks used by both structure and motion streams."""

    def __init__(self, num_node: int = 25, init_keep_bias: float = 0.5):
        super().__init__()
        node_init = torch.stack((torch.zeros(num_node), torch.full((num_node,), init_keep_bias)), dim=-1)
        edge_shape = (num_node, num_node)
        edge_init = torch.stack((torch.zeros(edge_shape), torch.full(edge_shape, init_keep_bias)), dim=-1)
        self.node_logits_1 = nn.Parameter(node_init.clone())
        self.node_logits_2 = nn.Parameter(node_init.clone())
        self.edge_logits_1 = nn.Parameter(edge_init.clone())
        self.edge_logits_2 = nn.Parameter(edge_init.clone())

    @staticmethod
    def _sample(logits: torch.Tensor, tau: float, hard: bool) -> torch.Tensor:
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)[..., 1]

    @staticmethod
    def _deterministic(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).to(dtype=logits.dtype)

    def forward(self, sample: bool | None = None, tau: float = 1.0, hard: bool = True) -> tuple[torch.Tensor, ...]:
        if sample is None:
            sample = self.training
        if sample:
            function = lambda value: self._sample(value, tau=tau, hard=hard)
        else:
            function = self._deterministic
        return (
            function(self.node_logits_1),
            function(self.node_logits_2),
            function(self.edge_logits_1),
            function(self.edge_logits_2),
        )

    @staticmethod
    def effective(masks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        node_1, node_2, edge_1, edge_2 = masks
        return node_1 * node_2, edge_1 * edge_2


class UnitTCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, 1),
            padding=(padding, 0), stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        _conv_init(self.conv)
        _bn_init(self.bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class UnitGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray, num_subset: int = 3):
        super().__init__()
        inter_channels = max(out_channels // 4, 1)
        self.inter_channels = inter_channels
        self.num_subset = num_subset
        self.register_buffer("A", torch.from_numpy(adjacency.astype(np.float32)))
        self.PA = nn.Parameter(torch.full_like(self.A, 1e-6))
        self.conv_a = nn.ModuleList([nn.Conv2d(in_channels, inter_channels, 1) for _ in range(num_subset)])
        self.conv_b = nn.ModuleList([nn.Conv2d(in_channels, inter_channels, 1) for _ in range(num_subset)])
        self.conv_d = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(num_subset)])
        self.down = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            if in_channels != out_channels else nn.Identity()
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                _bn_init(module)
        _bn_init(self.bn, 1e-6)

    def forward(
        self,
        x: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, channels, frames, joints = x.shape
        adjacency = self.A + self.PA
        if edge_mask is not None:
            adjacency = adjacency * edge_mask.view(1, joints, joints)
        if node_mask is not None:
            x = x * node_mask.view(1, 1, 1, joints)

        output: torch.Tensor | None = None
        for subset in range(self.num_subset):
            query = self.conv_a[subset](x).permute(0, 3, 1, 2).reshape(n, joints, -1)
            key = self.conv_b[subset](x).reshape(n, -1, joints)
            adaptive = torch.softmax(torch.matmul(query, key) / query.size(-1), dim=-2)
            adaptive = adaptive + adjacency[subset]
            values = x.reshape(n, channels * frames, joints)
            branch = torch.matmul(values, adaptive).reshape(n, channels, frames, joints)
            branch = self.conv_d[subset](branch)
            output = branch if output is None else output + branch
        assert output is not None
        return self.relu(self.bn(output) + self.down(x))


class TCN_GCN_Unit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray, stride: int = 1, residual: bool = True):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, adjacency)
        self.tcn = UnitTCN(out_channels, out_channels, stride=stride)
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, edge_mask: torch.Tensor | None = None, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        return self.relu(self.tcn(self.gcn(x, edge_mask=edge_mask, node_mask=node_mask)) + residual)


class InterFeatureSharingBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.transform = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, current: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        interaction = self.transform(torch.cat((self.bn(current), self.bn(other)), dim=1))
        return current + self.beta * interaction


class RCSGCN(nn.Module):
    """Seven-layer dual-stream RCS-GCN for 12-channel MG skeleton input."""

    def __init__(
        self,
        num_class: int = 4,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 12,
        dropout: float = 0.5,
    ):
        super().__init__()
        if (num_point, num_person, in_channels) != (25, 1, 12):
            raise ValueError("The public MG configuration requires num_point=25, num_person=1, in_channels=12.")
        adjacency = Graph(num_point).A
        layer_spec = ((6, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2),
                      (128, 128, 1), (128, 256, 2), (256, 256, 1))
        self.num_class = num_class
        self.num_person = num_person
        self.informative_mask = InformativeMask(num_point)
        self.structure_stream = nn.ModuleList([
            TCN_GCN_Unit(in_c, out_c, adjacency, stride=stride, residual=index != 0)
            for index, (in_c, out_c, stride) in enumerate(layer_spec)
        ])
        self.motion_stream = nn.ModuleList([
            TCN_GCN_Unit(in_c, out_c, adjacency, stride=stride, residual=index != 0)
            for index, (in_c, out_c, stride) in enumerate(layer_spec)
        ])
        self.share_l3_structure = InterFeatureSharingBlock(64)
        self.share_l3_motion = InterFeatureSharingBlock(64)
        self.share_l6_structure = InterFeatureSharingBlock(256)
        self.share_l6_motion = InterFeatureSharingBlock(256)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_class)
        self.recombined_classifier = nn.Linear(1024, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))
        nn.init.normal_(self.recombined_classifier.weight, 0, math.sqrt(2.0 / num_class))

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 5 or x.shape[1] != 12 or x.shape[3:] != (25, 1):
            raise ValueError(f"Expected (N,12,T,25,1), got {tuple(x.shape)}.")

    def forward_features(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_input(x)
        n, channels, frames, joints, people = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(n * people, channels, frames, joints)
        structure = x[:, :6]
        motion = x[:, 6:]
        for index, (structure_layer, motion_layer) in enumerate(zip(self.structure_stream, self.motion_stream)):
            structure = structure_layer(structure, edge_mask=edge_mask, node_mask=node_mask)
            motion = motion_layer(motion, edge_mask=edge_mask, node_mask=node_mask)
            if index == 2:
                structure, motion = (
                    self.share_l3_structure(structure, motion),
                    self.share_l3_motion(motion, structure),
                )
            if index == 5:
                structure, motion = (
                    self.share_l6_structure(structure, motion),
                    self.share_l6_motion(motion, structure),
                )
        structure = F.adaptive_avg_pool2d(structure, 1).reshape(n, people, -1).mean(dim=1)
        motion = F.adaptive_avg_pool2d(motion, 1).reshape(n, people, -1).mean(dim=1)
        return torch.cat((structure, motion), dim=1)

    def predict_whole(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.forward_features(x)))

    def predict_informative(self, x: torch.Tensor, masks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        node_mask, edge_mask = self.informative_mask.effective(masks)
        return self.classifier(self.dropout(self.forward_features(x, node_mask=node_mask, edge_mask=edge_mask)))

    def predict_noninformative(self, x: torch.Tensor, masks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        node_mask, edge_mask = self.informative_mask.effective(masks)
        return self.classifier(self.dropout(self.forward_features(x, node_mask=1.0 - node_mask, edge_mask=1.0 - edge_mask)))

    def predict_recombined(self, x: torch.Tensor, masks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        node_mask, edge_mask = self.informative_mask.effective(masks)
        informative = self.forward_features(x, node_mask=node_mask, edge_mask=edge_mask)
        noninformative = self.forward_features(x, node_mask=1.0 - node_mask, edge_mask=1.0 - edge_mask)
        noninformative = torch.roll(noninformative, shifts=1, dims=0)
        return self.recombined_classifier(self.dropout(torch.cat((informative, noninformative), dim=1)))

    def forward(self, x: torch.Tensor, use_informative_mask: bool = True) -> torch.Tensor:
        if not use_informative_mask:
            return self.predict_whole(x)
        masks = self.informative_mask(sample=self.training)
        return self.predict_informative(x, masks)


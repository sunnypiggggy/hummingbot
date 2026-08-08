"""Deterministic dual-resolution PyTorch models for the v12 research gate."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RobustSequenceScaler:
    median_hourly: np.ndarray
    iqr_hourly: np.ndarray
    median_five: np.ndarray
    iqr_five: np.ndarray

    @classmethod
    def fit(cls, hourly: np.ndarray, five: np.ndarray) -> "RobustSequenceScaler":
        def stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            flat = values.reshape(-1, values.shape[-1])
            median = np.nanmedian(flat, axis=0).astype(np.float32)
            q25, q75 = np.nanpercentile(flat, [25, 75], axis=0)
            iqr = (q75 - q25).astype(np.float32)
            iqr[~np.isfinite(iqr) | (iqr < 1e-8)] = 1.0
            median[~np.isfinite(median)] = 0.0
            return median, iqr

        mh, ih = stats(hourly)
        mf, iff = stats(five)
        return cls(mh, ih, mf, iff)

    @staticmethod
    def _transform(values: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
        output = (values.astype(np.float32) - median) / iqr
        output = np.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(output, -10.0, 10.0).astype(np.float32)

    def transform(self, hourly: np.ndarray, five: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._transform(hourly, self.median_hourly, self.iqr_hourly),
            self._transform(five, self.median_five, self.iqr_five),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "median_hourly": self.median_hourly.tolist(),
            "iqr_hourly": self.iqr_hourly.tolist(),
            "median_five": self.median_five.tolist(),
            "iqr_five": self.iqr_five.tolist(),
        }


def seed_everything(seed: int, threads: int = 2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(max(1, int(threads)))
    torch.use_deterministic_algorithms(True, warn_only=True)


class Chomp1d(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :, :-self.amount] if self.amount else values


class TCNEncoder(nn.Module):
    def __init__(self, inputs: int, hidden: int, dropout: float, layers: int = 4):
        super().__init__()
        blocks = []
        width = inputs
        for index in range(layers):
            dilation = 2 ** index
            padding = 2 * dilation
            blocks.extend([
                nn.Conv1d(width, hidden, kernel_size=3, dilation=dilation, padding=padding),
                Chomp1d(padding), nn.GELU(), nn.Dropout(dropout),
            ])
            width = hidden
        self.network = nn.Sequential(*blocks)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.transpose(1, 2))[:, :, -1]


class GRUEncoder(nn.Module):
    def __init__(self, inputs: int, hidden: int, dropout: float, layers: int):
        super().__init__()
        self.network = nn.GRU(
            inputs, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.network(values)
        return hidden[-1]


class TransformerEncoder(nn.Module):
    def __init__(self, inputs: int, hidden: int, dropout: float, layers: int):
        super().__init__()
        self.project = nn.Linear(inputs, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.network = nn.TransformerEncoder(layer, layers)
        self.attention = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.project(values)
        positions = torch.arange(encoded.shape[1], device=encoded.device, dtype=encoded.dtype)
        frequencies = torch.exp(
            torch.arange(0, encoded.shape[2], 2, device=encoded.device, dtype=encoded.dtype)
            * (-math.log(10000.0) / encoded.shape[2])
        )
        phase = positions[:, None] * frequencies[None, :]
        positional = torch.zeros_like(encoded[0])
        positional[:, 0::2] = torch.sin(phase)
        positional[:, 1::2] = torch.cos(phase[:, : positional[:, 1::2].shape[1]])
        encoded = self.network(encoded + positional.unsqueeze(0))
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        return torch.sum(encoded * weights.unsqueeze(-1), dim=1)


class DualBranchLongRiskModel(nn.Module):
    def __init__(self, hourly_features: int, five_features: int, config: Mapping[str, Any]):
        super().__init__()
        architecture = str(config["architecture"])
        # The 288 raw five-minute observations remain the input contract.  A
        # deterministic 15-minute mean stem reduces CPU cost before encoding.
        self.compress_five = True
        hidden = int(config["hidden"])
        dropout = float(config["dropout"])
        layers = int(config["layers"])
        encoder = {"tcn": TCNEncoder, "gru": GRUEncoder, "transformer": TransformerEncoder}[architecture]
        if architecture == "tcn":
            self.hourly = encoder(hourly_features, hidden, dropout, 4)
            self.five = encoder(five_features, hidden, dropout, 4)
        else:
            self.hourly = encoder(hourly_features, hidden, dropout, layers)
            self.five = encoder(five_features, hidden, dropout, layers)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, hourly: torch.Tensor, five: torch.Tensor) -> torch.Tensor:
        if self.compress_five:
            five = torch.nn.functional.avg_pool1d(
                five.transpose(1, 2), kernel_size=3, stride=3
            ).transpose(1, 2)
        return self.fusion(torch.cat([self.hourly(hourly), self.five(five)], dim=1))


def deterministic_configurations() -> list[dict[str, Any]]:
    anchors = {
        "tcn": [
            {"hidden": 32, "layers": 4, "dropout": .10, "learning_rate": 1e-3, "weight_decay": 1e-4},
            {"hidden": 64, "layers": 4, "dropout": .30, "learning_rate": 3e-4, "weight_decay": 1e-3},
        ],
        "gru": [
            {"hidden": 32, "layers": 1, "dropout": .10, "learning_rate": 1e-3, "weight_decay": 1e-4},
            {"hidden": 64, "layers": 2, "dropout": .30, "learning_rate": 3e-4, "weight_decay": 1e-3},
        ],
        "transformer": [
            {"hidden": 32, "layers": 2, "dropout": .10, "learning_rate": 1e-3, "weight_decay": 1e-4},
            {"hidden": 64, "layers": 3, "dropout": .30, "learning_rate": 3e-4, "weight_decay": 1e-3},
        ],
    }
    space = [
        {"hidden": hidden, "layers": layers, "dropout": dropout,
         "learning_rate": learning_rate, "weight_decay": weight_decay}
        for hidden in (32, 64) for layers in (1, 2, 3)
        for dropout in (.10, .20, .30) for learning_rate in (3e-4, 1e-3)
        for weight_decay in (1e-4, 1e-3)
    ]
    rng = np.random.default_rng(42)
    output = []
    for architecture in ("tcn", "gru", "transformer"):
        chosen = list(anchors[architecture])
        order = rng.permutation(len(space))
        for index in order:
            item = dict(space[int(index)])
            if architecture == "tcn":
                item["layers"] = 4
            if item not in chosen:
                chosen.append(item)
            if len(chosen) == 8:
                break
        for index, item in enumerate(chosen):
            output.append({
                "config_id": f"{architecture[:2]}{index:02d}",
                "architecture": architecture,
                "batch_size": {"tcn": 512, "gru": 256, "transformer": 128}[architecture],
                "max_epochs": 100, "patience": 10, **item,
            })
    if len(output) != 24 or len({item["config_id"] for item in output}) != 24:
        raise AssertionError("expected 24 deterministic deep-learning configurations")
    return output

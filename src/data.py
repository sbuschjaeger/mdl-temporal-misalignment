from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    dataset_id: str
    source: str
    sensor_groups: dict[str, list[int]]
    lags: list[int]
    n_folds: int = 5
    params: dict[str, Any] = field(default_factory=dict)
    data_root: Path | None = None

    def group_names(self) -> list[str]:
        return sorted(self.sensor_groups)

    def sensor_groups_indexed(self) -> dict[int, list[int]]:
        return {
            i: list(self.sensor_groups[name])
            for i, name in enumerate(self.group_names())
        }

    def group_index(self, name: str) -> int:
        return self.group_names().index(name)

    def aligned_channels(self, misalign_group: str) -> list[int]:
        out: list[int] = []
        for name in self.group_names():
            if name != misalign_group:
                out.extend(self.sensor_groups[name])
        return out


def load_dataset_config(path: str | Path, data_root: str | Path | None = None) -> DatasetConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    dataset_id = path.stem
    if "lags" not in data:
        raise ValueError(f"{path}: missing required field 'lags'")
    lags = [int(lag) for lag in data["lags"]]
    if not lags:
        raise ValueError(f"{path}: 'lags' must not be empty")
    if min(lags) < 0:
        raise ValueError(f"{path}: experiment lags must be non-negative")
    return DatasetConfig(
        dataset_id=dataset_id,
        source=data.get("monster_name") or data.get("synthetic") or dataset_id,
        n_folds=int(data.get("n_folds", 5)),
        sensor_groups={k: list(v) for k, v in data["sensor_groups"].items()},
        lags=lags,
        params=dict(data.get("params", {})),
        data_root=Path(data_root) if data_root is not None else None,
    )


def build_dataset(
    config: DatasetConfig,
    fold: int,
    split: str,
    *,
    lags: list[int],
    lag_group: int | None = None,
    lag: int = 0,
    normalize: bool = True,
    norm_stats=None,
):
    if min(int(lag) for lag in lags) < 0:
        raise ValueError("experiment lags must be non-negative")
    max_lag = max(int(lag) for lag in lags)
    return TimeSeriesDataset(
        config,
        fold=fold,
        split=split,
        max_lag=max_lag,
        lag_group=lag_group,
        lag=lag,
        normalize=normalize,
        norm_stats=norm_stats,
    )


class TimeSeriesDataset(Dataset):
    """Array-backed time-series dataset for MONSTER and synthetic sources."""

    def __init__(
        self,
        config: DatasetConfig,
        fold: int,
        split: str,
        max_lag: int,
        lag_group: int | None = None,
        lag: int = 0,
        normalize: bool = True,
        norm_stats=None,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if fold < 0:
            raise ValueError("fold must be non-negative")

        self.config = config
        self.dataset_name = config.source
        self.fold = fold
        self.split = split
        self.max_lag = int(max_lag)
        self.lag_group = lag_group
        self.lag = int(lag)
        if self.lag < 0:
            raise ValueError("experiment lags must be non-negative")
        if self.lag > self.max_lag:
            raise ValueError("lag must be covered by the dataset lag list")
        self.sensor_groups = config.sensor_groups_indexed()

        if config.source in {"syncsine_sensitive", "syncsine_robust"}:
            x_array, y_array, indices = make_syncsine_arrays(config, fold, split, self.max_lag)
        else:
            x_array, y_array, indices = self._download_and_load(self.dataset_name, fold, split)
        self._X = x_array
        self._indices = np.asarray(indices, dtype=np.int64)
        self._all_labels = np.asarray(y_array, dtype=np.int64)
        self._labels = self._all_labels[self._indices]

        self.raw_window_length = int(self._X.shape[2])
        if self.raw_window_length <= self.max_lag:
            raise ValueError("largest lag must be smaller than window length")
        self.window_size = self.raw_window_length - self.max_lag
        self.effective_length = self.window_size
        self.num_channels = int(self._X.shape[1])
        self.num_classes = int(np.max(y_array)) + 1
        self.window_labels = self._labels.tolist()

        self.normalize = normalize
        if normalize:
            if norm_stats is None:
                self.norm_mean, self.norm_std = self._compute_norm_stats()
            else:
                self.norm_mean = np.asarray(norm_stats[0], dtype=np.float32)
                self.norm_std = np.asarray(norm_stats[1], dtype=np.float32)
        else:
            self.norm_mean = None
            self.norm_std = None

    def _download_and_load(self, dataset_name: str, fold: int, split: str):
        if self.config.data_root is not None:
            dataset_dir = self.config.data_root / self.config.dataset_id
            shared_x_path = dataset_dir / "X.npy"
            fold_x_path = dataset_dir / f"fold_{fold}_X.npy"
            x_path = shared_x_path if shared_x_path.exists() else fold_x_path
            y_path = dataset_dir / "y.npy"
            test_idx_path = dataset_dir / f"test_indices_fold_{fold}.txt"
            local_paths = (x_path, y_path, test_idx_path)
            if all(path.exists() for path in local_paths):
                x = np.load(x_path, mmap_mode="r")
                y = np.load(y_path)
                test_indices = np.loadtxt(test_idx_path, dtype=np.int64, ndmin=1)
                if split == "test":
                    return x, y, test_indices
                mask = np.zeros(len(x), dtype=bool)
                mask[test_indices] = True
                return x, y, np.where(~mask)[0].astype(np.int64)

        from huggingface_hub import hf_hub_download

        repo_id = f"monster-monash/{dataset_name}"
        x_path = hf_hub_download(
            repo_id=repo_id, filename=f"{dataset_name}_X.npy", repo_type="dataset"
        )
        y_path = hf_hub_download(
            repo_id=repo_id, filename=f"{dataset_name}_y.npy", repo_type="dataset"
        )
        test_idx_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"test_indices_fold_{fold}.txt",
            repo_type="dataset",
        )

        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path)
        with open(test_idx_path) as fh:
            test_indices = np.array([int(line) for line in fh if line.strip()], dtype=np.int64)

        if split == "test":
            return x, y, test_indices
        mask = np.zeros(len(x), dtype=bool)
        mask[test_indices] = True
        return x, y, np.where(~mask)[0].astype(np.int64)

    def _compute_norm_stats(self):
        total = np.zeros(self.num_channels, dtype=np.float64)
        total_sq = np.zeros(self.num_channels, dtype=np.float64)
        n_values = 0
        for i in self._indices:
            sample = np.asarray(self._X[i], dtype=np.float64)
            total += sample.sum(axis=1)
            total_sq += (sample ** 2).sum(axis=1)
            n_values += sample.shape[1]
        mean = total / n_values
        var = total_sq / n_values - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return x
        return (x - self.norm_mean[:, None]) / self.norm_std[:, None]

    def __len__(self) -> int:
        return len(self._indices)

    def make_window(self, idx: int, group_id: int | None = None, lag: int = 0) -> torch.Tensor:
        full = self._X[self._indices[idx]]
        lag = int(lag)
        if lag < 0:
            raise ValueError("experiment lags must be non-negative")
        if lag > self.max_lag:
            raise ValueError("lag must be covered by the dataset lag list")
        # The complement is the fixed reference. A positive lag delays only
        # the selected group while preserving the same overlap length.
        out = np.array(full[:, : self.window_size], dtype=np.float32, copy=True)
        if group_id is not None and lag > 0:
            channels = self.sensor_groups[int(group_id)]
            out[channels, :] = full[channels, lag : lag + self.window_size]
        return torch.from_numpy(self._normalize(out))

    def __getitem__(self, idx: int):
        return self.make_window(idx, self.lag_group, self.lag), int(self._labels[idx])

    def subset(self, positions, *, normalize: bool | None = None, norm_stats=None):
        clone = object.__new__(TimeSeriesDataset)
        clone.__dict__ = self.__dict__.copy()
        positions = np.asarray(positions, dtype=np.int64)
        clone._indices = self._indices[positions]
        clone._labels = self._labels[positions]
        clone.window_labels = clone._labels.tolist()

        clone.normalize = self.normalize if normalize is None else bool(normalize)
        if clone.normalize:
            if norm_stats is None:
                clone.norm_mean, clone.norm_std = clone._compute_norm_stats()
            else:
                clone.norm_mean = np.asarray(norm_stats[0], dtype=np.float32)
                clone.norm_std = np.asarray(norm_stats[1], dtype=np.float32)
        else:
            clone.norm_mean = None
            clone.norm_std = None
        return clone

    def get_norm_stats(self):
        if self.norm_mean is None:
            return None
        return self.norm_mean.copy(), self.norm_std.copy()

def stratified_folds(labels: np.ndarray, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(labels), dtype=np.int64)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        assignment[idx] = np.arange(len(idx)) % n_folds
    return [np.where(assignment == fold)[0] for fold in range(n_folds)]


def make_syncsine_arrays(config: DatasetConfig, fold: int, split: str, max_lag: int):
    p = config.params
    seed = int(p["seed"])
    n_windows = int(p["n_windows"])
    window_size = int(p["window_size"])
    total_time = window_size + int(max_lag)
    dt = float(p.get("dt", 0.01))
    tau = float(p.get("tau", 1.0))
    r = float(p.get("r", 1.0))
    noise_std = float(p.get("noise_std", 0.05))
    bias_std = float(p.get("bias_std", 0.7))

    rng = np.random.default_rng(seed)
    t = np.arange(total_time, dtype=np.float32) * dt
    x = np.empty((n_windows, 2, total_time), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.int64)
    robust_labels = None
    if config.source == "syncsine_robust":
        robust_labels = np.arange(n_windows, dtype=np.int64) % 2
        rng.shuffle(robust_labels)

    for i in range(n_windows):
        if config.source == "syncsine_sensitive":
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            clean = np.sin(t / r + phase)
            bias0 = float(rng.normal(0.0, bias_std))
            bias1 = float(rng.normal(0.0, bias_std))
            ch0 = clean + bias0 + rng.normal(0.0, noise_std, size=total_time)
            ch1 = clean + bias1 + rng.normal(0.0, noise_std, size=total_time)
            y[i] = int((ch0[:window_size].mean() + ch1[:window_size].mean()) > tau)
        else:
            low_range = p.get("low_frequency_range", (2.0, 4.5))
            high_range = p.get("high_frequency_range", (5.5, 8.0))
            nuisance_range = p.get("nuisance_frequency_range", (2.0, 8.0))
            amplitude_range = p.get("amplitude_range", (0.8, 1.2))
            label = int(robust_labels[i])
            frequency_range = high_range if label else low_range
            f0 = float(rng.uniform(*frequency_range))
            f1 = float(rng.uniform(*nuisance_range))
            phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
            phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
            amplitude0 = float(rng.uniform(*amplitude_range))
            amplitude1 = float(rng.uniform(*amplitude_range))
            ch0 = amplitude0 * np.sin(2.0 * np.pi * f0 * t + phase0)
            ch1 = amplitude1 * np.sin(2.0 * np.pi * f1 * t + phase1)
            ch0 += float(rng.normal(0.0, bias_std)) + rng.normal(0.0, noise_std, size=total_time)
            ch1 += float(rng.normal(0.0, bias_std)) + rng.normal(0.0, noise_std, size=total_time)
            y[i] = label
        x[i, 0, :] = ch0
        x[i, 1, :] = ch1

    folds = stratified_folds(y, config.n_folds, seed)
    test_idx = folds[fold]
    if split == "test":
        indices = test_idx
    else:
        mask = np.ones(n_windows, dtype=bool)
        mask[test_idx] = False
        indices = np.where(mask)[0]
    return x, y, np.sort(indices).astype(np.int64)

class LaggedTrainingDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        group_id: int,
        lags: list[int],
        p_misalign: float = 1.0,
        seed: int = 42,
        return_pair: bool = False,
        deterministic: bool = False,
    ):
        self.base_dataset = base_dataset
        self.group_id = int(group_id)
        self.lags = [int(lag) for lag in lags if int(lag) != 0]
        if any(lag < 0 for lag in self.lags):
            raise ValueError("SyncNet training lags must be positive")
        self.p_misalign = float(p_misalign)
        self.rng = np.random.default_rng(seed)
        self.return_pair = return_pair
        self.deterministic = bool(deterministic)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        if self.return_pair:
            positive, _ = self.base_dataset[idx]
            lag = self.lags[idx % len(self.lags)] if self.deterministic else int(self.rng.choice(self.lags))
            negative = self.base_dataset.make_window(idx, self.group_id, lag)
            return positive, negative

        if self.lags and self.rng.random() < self.p_misalign:
            lag = int(self.rng.choice(self.lags))
            return self.base_dataset.make_window(idx, self.group_id, lag), int(self.base_dataset.window_labels[idx])
        return self.base_dataset[idx]


def windows_numpy(dataset):
    xs, ys = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x.numpy())
        ys.append(y)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def raw_windows_numpy(dataset):
    x = np.asarray(dataset._X[dataset._indices], dtype=np.float32)
    y = np.asarray(dataset._labels, dtype=np.int64)
    return x, y

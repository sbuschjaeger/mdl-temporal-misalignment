from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score

from src.data import LaggedTrainingDataset, build_dataset, raw_windows_numpy
from src.metric import _compute_smdl_arrays
from src.models import make_model


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*TreeSpec.*")

CLASSIFIER_MODELS = {"cnn", "resnet", "late_fusion", "magnitude_cnn"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_val_positions(labels, val_frac: float, seed: int):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * float(val_frac)))
        if len(cls_idx) > 1:
            n_val = min(len(cls_idx) - 1, max(1, n_val))
        else:
            n_val = 0
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    return np.sort(train_idx).astype(np.int64), np.sort(val_idx).astype(np.int64)


def classifier_accuracy(model, dataset, device, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            correct += int((pred == y).sum())
            total += int(y.numel())
    return correct / total


class ClassifierModule(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, lr: float, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = make_model(model_name, num_classes=num_classes, **model_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits.argmax(dim=1), y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        pred = logits.argmax(dim=1)
        self.val_acc(pred, y)
        self.val_f1(pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

class SyncNetModule(pl.LightningModule):
    def __init__(self, lr: float, margin: float = 1.0, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = make_model("syncnet", num_classes=2, in_channels=0, **model_kwargs)

    def forward(self, x):
        return self.model(x)

    def contrastive_loss(self, pos_a, pos_b, neg_a, neg_b):
        dist_pos = torch.norm(pos_a - pos_b, dim=1)
        dist_neg = torch.norm(neg_a - neg_b, dim=1)
        loss = F.relu(dist_pos - dist_neg + float(self.hparams.margin)).mean()
        return loss, dist_pos, dist_neg

    def training_step(self, batch, batch_idx):
        pos, neg = batch
        pos_a, pos_b = self.model(pos)
        neg_a, neg_b = self.model(neg)
        loss, dist_pos, dist_neg = self.contrastive_loss(pos_a, pos_b, neg_a, neg_b)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_dist_pos", dist_pos.mean(), on_epoch=True)
        self.log("train_dist_neg", dist_neg.mean(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pos, neg = batch
        pos_a, pos_b = self.model(pos)
        neg_a, neg_b = self.model(neg)
        loss, dist_pos, dist_neg = self.contrastive_loss(pos_a, pos_b, neg_a, neg_b)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_dist_pos", dist_pos.mean(), on_epoch=True)
        self.log("val_dist_neg", dist_neg.mean(), on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=float(self.hparams.lr), weight_decay=5e-4)

def train_one_model(
    config,
    model_name: str,
    fold: int,
    lags: list[int],
    training: dict,
    output_dir: Path,
    seed: int,
    group_name: str | None = None,
    reuse_checkpoint: bool = True,
):
    set_seed(seed)
    normalize = bool(training.get("normalize", True))
    val_frac = float(training.get("val_frac", 0.15))
    full_train = build_dataset(config, fold, "train", lags=lags, normalize=False)
    train_idx, val_idx = train_val_positions(full_train.window_labels, val_frac, seed)
    train_base = full_train.subset(train_idx, normalize=normalize)
    val_base = full_train.subset(
        val_idx,
        normalize=normalize,
        norm_stats=train_base.get_norm_stats(),
    )

    misalign_group_name = group_name or config.group_names()[0]
    group_id = config.group_index(misalign_group_name)
    run_name = model_name if model_name != "syncnet" else f"syncnet_{misalign_group_name}"
    model_dir = output_dir / config.dataset_id / run_name / f"fold_{fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(model_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if reuse_checkpoint and existing and (model_dir / "norm_stats.pt").exists():
        return str(existing[0])

    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))

    if model_name == "syncnet":
        module = SyncNetModule(
            lr=float(training.get("lr", 1e-4)),
            window_size=train_base.window_size,
            misalign_indices=config.sensor_groups[misalign_group_name],
            aligned_indices=config.aligned_channels(misalign_group_name),
        )
        train_ds = LaggedTrainingDataset(train_base, group_id, lags=lags, seed=seed, return_pair=True)
        val_ds = LaggedTrainingDataset(
            val_base,
            group_id,
            lags=lags,
            seed=seed + 1,
            return_pair=True,
            deterministic=True,
        )
    else:
        module = ClassifierModule(
            model_name=model_name,
            num_classes=train_base.num_classes,
            lr=float(training.get("lr", 1e-4)),
            in_channels=train_base.num_channels,
            sensor_groups=train_base.sensor_groups,
            window_size=train_base.window_size,
        )
        p_misalign = float(training.get("p_misalign", 0.0))
        train_ds = LaggedTrainingDataset(train_base, group_id, lags, p_misalign, seed=seed) if p_misalign > 0 else train_base
        val_ds = val_base

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    monitor = "val_loss"
    monitor_mode = "min"
    checkpoint_name = "best_val_loss={val_loss:.4f}"
    checkpoint = ModelCheckpoint(
        dirpath=str(model_dir),
        filename=checkpoint_name,
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor=monitor,
        mode=monitor_mode,
        patience=int(training.get("patience", 10)),
    )
    trainer_kwargs = {
        "max_epochs": int(training.get("max_epochs", 100)),
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "logger": False,
        "callbacks": [checkpoint, early_stop],
        "enable_progress_bar": bool(training.get("progress_bar", False)),
        "enable_model_summary": bool(training.get("model_summary", False)),
    }
    if torch.cuda.is_available():
        trainer_kwargs["precision"] = training.get("precision", "32-true")
    for key in ("limit_train_batches", "limit_val_batches", "fast_dev_run"):
        if key in training:
            trainer_kwargs[key] = training[key]
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, train_loader, val_loader)
    torch.save(train_base.get_norm_stats(), model_dir / "norm_stats.pt")
    return checkpoint.best_model_path or ""

def evaluate_classifier_lags(
    config,
    ckpt_path: str,
    model_name: str,
    fold: int,
    lags: list[int],
    training: dict,
    batch_size: int,
    group_names: list[str] | None = None,
):
    normalize = bool(training.get("normalize", True))
    norm_stats = torch.load(Path(ckpt_path).parent / "norm_stats.pt", weights_only=False)
    module = ClassifierModule.load_from_checkpoint(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.model.to(device).eval()

    rows = []
    with torch.no_grad():
        for group_name in group_names or config.group_names():
            group_id = config.group_index(group_name)
            for lag in lags:
                test_ds = build_dataset(
                    config,
                    fold,
                    "test",
                    lags=lags,
                    lag_group=group_id,
                    lag=int(lag),
                    normalize=normalize,
                    norm_stats=norm_stats,
                )
                rows.append({
                    "dataset": config.dataset_id,
                    "fold": fold,
                    "model": model_name,
                    "group": group_name,
                    "lag": int(lag),
                    "accuracy": classifier_accuracy(module.model, test_ds, device, batch_size),
                })
    return pd.DataFrame(rows)


def compute_correction_curve(
    config,
    fold: int,
    group_name: str,
    total_lags: list[int],
    training: dict,
    seed: int,
):
    """Compute the MDL correction curve directly on the deployment test set."""
    test_ds = build_dataset(
        config,
        fold,
        "test",
        lags=total_lags,
        normalize=False,
    )
    x_test, _ = raw_windows_numpy(test_ds)
    _, group_summary, lag_curves = _compute_smdl_arrays(
        config,
        x_test,
        None,
        total_lags,
        training.get("correction_max_samples"),
        seed,
        group_names=[group_name],
        fixed_window_size=test_ds.window_size,
        circular=False,
    )
    row = group_summary[group_summary["group_name"] == group_name].iloc[0]
    curve = lag_curves[lag_curves["group_name"] == group_name].copy()
    return {
        "best_lag": int(row["best_lag"]),
        "gain_at_zero_bits": float(row["gain_at_zero_bits"]),
        "maximum_gain_bits": float(row["maximum_gain_bits"]),
        "gain_range_bits": float(row["gain_range_bits"]),
        "zero_is_best": bool(row["zero_is_best"]),
        "curve": curve,
    }


def select_mdl_correction(curve, imposed_lag: int, correction_lags: list[int]):
    """Choose the correction minimizing raw MDL alignment cost."""
    candidates = pd.DataFrame({
        "predicted_correction": [int(lag) for lag in correction_lags],
    })
    candidates["corrected_lag"] = int(imposed_lag) + candidates["predicted_correction"]
    candidates = candidates.merge(
        curve[["tau", "gain_bits", "alignment_cost_bits"]],
        left_on="corrected_lag",
        right_on="tau",
        how="inner",
    )
    if candidates.empty:
        raise ValueError("no correction candidate is covered by the MDL curve")
    candidates = candidates.sort_values(
        ["alignment_cost_bits", "predicted_correction"],
        key=lambda values: values.abs() if values.name == "predicted_correction" else values,
    )
    best = candidates.iloc[0]
    return {
        "predicted_correction": int(best["predicted_correction"]),
        "corrected_lag": int(best["corrected_lag"]),
        "gain_after_bits": float(best["gain_bits"]),
        "cost_after_bits": float(best["alignment_cost_bits"]),
        "abstained": False,
    }


def dense_correction_grid(lags: list[int]) -> list[int]:
    """Correct a positive imposed delay only back toward nominal alignment."""
    ordered = np.asarray(sorted(set(int(lag) for lag in lags)), dtype=int)
    if len(ordered) == 1:
        return [0]
    step = int(np.gcd.reduce(np.diff(ordered)))
    maximum = int(np.max(ordered))
    return list(range(-maximum, 1, step))


def evaluate_classifier_corrections(
    config,
    ckpt_path: str,
    model_name: str,
    fold: int,
    lags: list[int],
    training: dict,
    batch_size: int,
    correction_cache: dict,
    seed: int,
    group_names: list[str] | None = None,
    imposed_lags: list[int] | None = None,
):
    normalize = bool(training.get("normalize", True))
    norm_stats = torch.load(Path(ckpt_path).parent / "norm_stats.pt", weights_only=False)
    module = ClassifierModule.load_from_checkpoint(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.model.to(device).eval()
    max_lag = max(int(lag) for lag in lags)

    rows = []
    group_names = group_names or config.group_names()
    imposed_lags = imposed_lags or [int(lag) for lag in lags]
    unknown_lags = set(imposed_lags) - set(lags)
    if unknown_lags:
        raise ValueError(f"correction lags are not in the dataset lag grid: {sorted(unknown_lags)}")
    correction_lags = dense_correction_grid(lags)
    total_lags = sorted({
        int(imposed) + int(correction)
        for imposed in imposed_lags
        for correction in correction_lags
        if 0 <= int(imposed) + int(correction) <= max_lag
    } | {0})
    for group_name in group_names:
        group_id = config.group_index(group_name)
        baseline_ds = build_dataset(
            config,
            fold,
            "test",
            lags=lags,
            lag_group=group_id,
            lag=0,
            normalize=normalize,
            norm_stats=norm_stats,
        )
        accuracy_by_lag = {
            0: classifier_accuracy(module.model, baseline_ds, device, batch_size)
        }

        def accuracy_at(lag):
            lag = int(lag)
            if lag not in accuracy_by_lag:
                dataset = build_dataset(
                    config,
                    fold,
                    "test",
                    lags=lags,
                    lag_group=group_id,
                    lag=lag,
                    normalize=normalize,
                    norm_stats=norm_stats,
                )
                accuracy_by_lag[lag] = classifier_accuracy(
                    module.model, dataset, device, batch_size
                )
            return accuracy_by_lag[lag]

        baseline_accuracy = accuracy_by_lag[0]

        for imposed_lag in imposed_lags:
            lagged_accuracy = accuracy_at(imposed_lag)

            cache_key = (config.dataset_id, fold, group_name)
            if cache_key not in correction_cache:
                correction_cache[cache_key] = compute_correction_curve(
                    config, fold, group_name, total_lags, training, seed,
                )
            correction = correction_cache[cache_key]
            before = correction["curve"][
                correction["curve"]["tau"] == int(imposed_lag)
            ].iloc[0]
            selected = select_mdl_correction(
                correction["curve"], int(imposed_lag), correction_lags
            )
            corrected_lag = selected["corrected_lag"]
            corrected_accuracy = accuracy_at(corrected_lag)

            rows.append({
                "dataset": config.dataset_id,
                "fold": fold,
                "model": model_name,
                "group": group_name,
                "metric": "unconditional",
                "imposed_lag": int(imposed_lag),
                "estimated_correction": selected["predicted_correction"],
                "corrected_lag": corrected_lag,
                "alignment_error": abs(corrected_lag),
                "gain_before_bits": float(before["gain_bits"]),
                "gain_after_bits": selected["gain_after_bits"],
                "cost_before_bits": float(before["alignment_cost_bits"]),
                "cost_after_bits": selected["cost_after_bits"],
                "maximum_gain_bits": correction["maximum_gain_bits"],
                "gain_range_bits": correction["gain_range_bits"],
                "abstained": selected["abstained"],
                "zero_is_best": correction["zero_is_best"],
                "baseline_accuracy": baseline_accuracy,
                "lagged_accuracy": lagged_accuracy,
                "corrected_accuracy": corrected_accuracy,
                "accuracy_improvement": corrected_accuracy - lagged_accuracy,
            })
    return pd.DataFrame(rows)

def evaluate_syncnet_lags(
    config,
    ckpt_path: str,
    fold: int,
    group_name: str,
    lags: list[int],
    training: dict,
    batch_size: int,
    split: str = "test",
):
    normalize = bool(training.get("normalize", True))
    norm_stats = torch.load(Path(ckpt_path).parent / "norm_stats.pt", weights_only=False)
    module = SyncNetModule.load_from_checkpoint(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.model.to(device).eval()
    rows = []
    with torch.no_grad():
        group_id = config.group_index(group_name)
        for lag in lags:
            dists = []
            test_ds = build_dataset(
                config,
                fold,
                split,
                lags=lags,
                lag_group=group_id,
                lag=int(lag),
                normalize=normalize,
                norm_stats=norm_stats,
            )
            loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            for x, _ in loader:
                a, b = module.model(x.to(device))
                dists.append(torch.norm(a - b, dim=1).cpu().numpy())
            rows.append({
                "dataset": config.dataset_id,
                "fold": fold,
                "model": "syncnet",
                "group": group_name,
                "split": split,
                "lag": int(lag),
                "mean_distance": float(np.concatenate(dists).mean()),
            })
    return pd.DataFrame(rows)

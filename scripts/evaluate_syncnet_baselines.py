#!/usr/bin/env python
"""Evaluate existing SyncNet checkpoints on their unshifted training folds."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from tqdm import tqdm

from src.data import load_dataset_config
from src.train import evaluate_syncnet_lags


def latest_checkpoint(results_dir: Path, dataset: str, group: str, fold: int) -> Path:
    model_dir = results_dir / "checkpoints" / dataset / f"syncnet_{group}" / f"fold_{fold}"
    checkpoints = sorted(
        model_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"no SyncNet checkpoint in {model_dir}")
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--results-dir", default="results/original")
    parser.add_argument("--output-file")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output_file or results_dir / "syncnet_train_lag_distance.csv")
    all_rows = []

    for config_path in tqdm(args.configs, desc="SyncNet training baselines", unit="dataset"):
        config_path = Path(config_path)
        exp = yaml.safe_load(config_path.read_text())
        config = load_dataset_config(config_path)
        selected = pd.read_csv(results_dir / f"{config.dataset_id}_selected_group.csv").iloc[0]
        group = str(selected["group"])
        training = dict(exp.get("training", {}))
        batch_size = int(exp.get("eval_batch_size", training.get("batch_size", 256)))

        for fold in range(config.n_folds):
            checkpoint = latest_checkpoint(results_dir, config.dataset_id, group, fold)
            rows = evaluate_syncnet_lags(
                config,
                str(checkpoint),
                fold,
                group,
                config.lags,
                training,
                batch_size,
                split="train",
            )
            all_rows.append(rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(output_file, index=False)
    print(f"wrote {output_file}", flush=True)


if __name__ == "__main__":
    main()

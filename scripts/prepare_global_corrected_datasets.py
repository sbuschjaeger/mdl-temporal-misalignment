#!/usr/bin/env python
"""Materialize one label-free whole-dataset MDL correction per real dataset.

The selected group is the group with the largest mean alignment cost, matching
the aggregation described in the paper. Its minimum-cost lag is applied once to
the complete array before the original cross-validation splits are reused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from src.data import load_dataset_config


REAL_DATASETS = [
    "crowdsourced", "dreamera", "dreamerv", "fordchallenge", "opportunity",
    "pamap2", "skoda", "stew", "uciactivity",
]


def monster_file(source: str, filename: str) -> Path:
    return Path(hf_hub_download(
        repo_id=f"monster-monash/{source}", filename=filename, repo_type="dataset",
    ))


def select_global_alignment(group_summary: pd.DataFrame, dataset: str) -> pd.Series:
    rows = group_summary[group_summary["dataset"] == dataset]
    if rows.empty:
        raise ValueError(f"no whole-dataset metric rows for {dataset}")
    # This is the same worst-group aggregation used by mdl_alignment_loss.
    return rows.loc[rows["mean_alignment_cost_bits"].idxmax()]


def write_global_correction(
    source_path: Path,
    output_path: Path,
    channels: list[int],
    lag: int,
    window_size: int,
    chunk_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source = np.load(source_path, mmap_mode="r")
    if lag + window_size > source.shape[-1]:
        raise ValueError(f"lag {lag} does not fit {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=source.dtype,
        shape=(*source.shape[:-1], window_size),
    )
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        output[start:stop] = source[start:stop, :, :window_size]
        if lag > 0:
            output[start:stop, channels] = source[
                start:stop, channels, lag : lag + window_size
            ]
    output.flush()
    return tuple(source.shape), tuple(output.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-results", default="results/global_audit")
    parser.add_argument("--data-root", default="data/global_corrected")
    parser.add_argument("--datasets", default=",".join(REAL_DATASETS))
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    audit_results = Path(args.audit_results)
    data_root = Path(args.data_root)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    summary_path = audit_results / "unconditional" / "group_summary.csv"
    group_summary = pd.read_csv(summary_path)
    manifest = []

    for dataset in datasets:
        config = load_dataset_config(Path("configs") / f"{dataset}.yml")
        selected = select_global_alignment(group_summary, dataset)
        group = str(selected["group_name"])
        lag = int(selected["best_lag"])
        maximum_lag = max(config.lags)
        window_size = None

        x_path = monster_file(config.source, f"{config.source}_X.npy")
        y_path = monster_file(config.source, f"{config.source}_y.npy")
        source = np.load(x_path, mmap_mode="r")
        window_size = int(source.shape[-1]) - maximum_lag
        original_shape, corrected_shape = write_global_correction(
            x_path,
            data_root / dataset / "X.npy",
            config.sensor_groups[group],
            lag,
            window_size,
            args.chunk_size,
        )
        np.save(data_root / dataset / "y.npy", np.load(y_path))
        for fold in range(config.n_folds):
            test_path = monster_file(config.source, f"test_indices_fold_{fold}.txt")
            (data_root / dataset / f"test_indices_fold_{fold}.txt").write_text(
                test_path.read_text()
            )

        manifest.append({
            "dataset": dataset,
            "group": group,
            "global_lag": lag,
            "at_boundary": bool(lag > 0 and lag == maximum_lag),
            "maximum_lag": maximum_lag,
            "cost_at_zero_bits": float(selected["cost_at_zero_bits"]),
            "mean_alignment_cost_bits": float(selected["mean_alignment_cost_bits"]),
            "original_length": original_shape[-1],
            "model_window_length": corrected_shape[-1],
        })
        print(
            f"global correction dataset={dataset} group={group} lag={lag} "
            f"boundary={lag == maximum_lag} length={original_shape[-1]}->{corrected_shape[-1]}",
            flush=True,
        )

    manifest_frame = pd.DataFrame(manifest)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(data_root / "manifest.csv", index=False)
    (data_root / "metadata.json").write_text(json.dumps({
        "audit_results": str(audit_results),
        "metric_summary": str(summary_path),
        "datasets": datasets,
        "rule": "whole-dataset unconditional MDL worst group and best lag",
        "classifier_evaluation": "five-fold CV at zero additional lag",
    }, indent=2))
    print(manifest_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

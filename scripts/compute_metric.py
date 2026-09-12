#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.data import load_dataset_config
from src.metric import compute_smdl_for_dataset


def parse_folds(value, n_folds):
    if value == "all":
        return list(range(n_folds))
    return [int(part) for part in value.split(",") if part.strip()]


def comma_ints(value):
    return None if value is None else [int(part) for part in value.split(",") if part.strip()]


def run_jobs(
    jobs,
    output_dir: Path,
    split: str,
    seed: int,
    max_samples,
    with_labels: bool,
    metadata,
    data_root=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    all_group_rows = []
    all_lag_rows = []
    for dataset_id, config_path, folds, lags in jobs:
        ds_config = load_dataset_config(config_path, data_root=data_root)
        for fold in folds:
            print(f"smdl dataset={dataset_id} fold={fold} split={split} lags={lags}", flush=True)
            result, group_summary, lag_curves = compute_smdl_for_dataset(
                ds_config,
                fold=fold,
                split=split,
                lags=lags,
                max_samples=max_samples,
                seed=seed + fold,
                with_labels=with_labels,
            )
            group_summary.insert(0, "dataset", dataset_id)
            group_summary.insert(1, "fold", fold)
            lag_curves.insert(0, "dataset", dataset_id)
            lag_curves.insert(1, "fold", fold)
            all_group_rows.append(group_summary)
            all_lag_rows.append(lag_curves)

            ds_dir = output_dir / dataset_id
            ds_dir.mkdir(exist_ok=True)
            (ds_dir / f"fold_{fold}_result.json").write_text(json.dumps(result, indent=2, default=str))

    if all_group_rows:
        pd.concat(all_group_rows, ignore_index=True).to_csv(output_dir / "group_summary.csv", index=False)
    if all_lag_rows:
        pd.concat(all_lag_rows, ignore_index=True).to_csv(output_dir / "lag_curves.csv", index=False)


def run_configs(config_paths, args):
    configs = [(Path(path), yaml.safe_load(Path(path).read_text())) for path in config_paths]
    configured_data_roots = {
        exp.get("data_root") for _, exp in configs if exp.get("data_root")
    }
    data_root = args.data_root
    if data_root is None and configured_data_roots:
        if len(configured_data_roots) != 1:
            raise ValueError("all metric configs must use the same data root")
        data_root = configured_data_roots.pop()
    seeds = {int(exp.get("seed", 42)) for _, exp in configs}
    if len(seeds) != 1:
        raise ValueError("all metric configs must use the same seed")

    override_lags = comma_ints(args.lags)
    jobs = []
    for config_path, exp in configs:
        ds_config = load_dataset_config(config_path, data_root=data_root)
        folds = parse_folds(args.folds or str(exp.get("folds", "all")), ds_config.n_folds)
        jobs.append((
            ds_config.dataset_id,
            config_path,
            folds,
            override_lags or ds_config.lags,
        ))

    results_dir = Path(args.results_dir)

    seed = seeds.pop()
    for variant_name, with_labels in (("unconditional", False), ("conditional", True)):
        variant_output = results_dir / variant_name
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": [str(path) for path, _ in configs],
            "metric_variant": variant_name,
            "lags_source": "--lags" if override_lags else "dataset config",
            "split": args.split,
            "max_samples": args.max_samples,
            "labels_used": with_labels,
        }
        run_jobs(
            jobs,
            variant_output,
            args.split,
            seed,
            args.max_samples,
            with_labels,
            metadata,
            data_root,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--results-dir", default="results/original")
    parser.add_argument("--folds", help="comma-separated fold override, or 'all'")
    parser.add_argument("--lags", help="comma-separated non-negative lag override")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    run_configs(args.configs, args)


if __name__ == "__main__":
    main()

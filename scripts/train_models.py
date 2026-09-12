#!/usr/bin/env python
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
from src.train import (
    CLASSIFIER_MODELS,
    evaluate_classifier_corrections,
    evaluate_classifier_lags,
    evaluate_syncnet_lags,
    train_one_model,
)


def parse_folds(value, n_folds):
    if value == "all":
        return list(range(n_folds))
    return [int(part) for part in value.split(",") if part.strip()]


def comma_list(value):
    return None if value is None else [part.strip() for part in value.split(",") if part.strip()]


def select_syncnet_group_from_frames(frames, dataset_id: str) -> str:
    if not frames:
        raise ValueError(f"no classifier lag results available for {dataset_id!r}")
    results = pd.concat(frames, ignore_index=True)
    keys = ["fold", "model", "group"] if "model" in results else ["fold", "group"]
    baseline = results[results["lag"] == 0].groupby(keys)["accuracy"].mean()
    shifted = results[results["lag"] != 0].groupby(keys)["accuracy"].min()
    drops = baseline.subtract(shifted).dropna().groupby("group").mean()
    if drops.empty:
        raise ValueError(f"cannot calculate an accuracy drop for {dataset_id!r}")
    return str(drops.idxmax())


def select_syncnet_group(results_dir: Path, dataset_id: str) -> str:
    frames = []
    for path in sorted(results_dir.glob(f"{dataset_id}_*_lag_accuracy.csv")):
        frame = pd.read_csv(path)
        if {"fold", "group", "lag", "accuracy"}.issubset(frame.columns):
            frames.append(frame)
    return select_syncnet_group_from_frames(frames, dataset_id)


def run_config(config_path: Path, args):
    exp = yaml.safe_load(config_path.read_text())
    results_dir = Path(args.results_dir)
    data_root = args.data_root or exp.get("data_root")
    training = dict(exp.get("training", {}))
    eval_batch_size = int(exp.get("eval_batch_size", training.get("batch_size", 256)))
    seed = int(exp.get("seed", 42))
    # The paper evaluates one deployment correction: the unconditional,
    # label-free PCA metric. Labels are deliberately not available here.
    evaluate_correction = not args.skip_corrections

    models = comma_list(args.models) or list(exp["models"])
    unknown_models = set(models) - set(exp["models"])
    if unknown_models:
        raise ValueError(f"unknown models: {sorted(unknown_models)}")
    eval_rows = {}
    correction_rows = {}
    correction_cache = {}
    syncnet_rows = {}
    selected_groups = {}

    ds_config = load_dataset_config(config_path, data_root=data_root)
    folds = parse_folds(args.folds or str(exp.get("folds", "all")), ds_config.n_folds)
    override_lags = (
        [int(lag) for lag in comma_list(args.lags)] if args.lags else None
    )
    fixed_selected_group = (
        select_syncnet_group(Path(args.fixed_groups_from), ds_config.dataset_id)
        if args.fixed_groups_from else None
    )
    classifier_models = [model for model in models if model in CLASSIFIER_MODELS]
    unknown = set(models) - CLASSIFIER_MODELS - {"syncnet"}
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    dataset_jobs = [(ds_config, folds, classifier_models)]

    n_training_jobs = sum(
        len(folds) * (len(classifier_models) + int("syncnet" in models))
        for _, folds, classifier_models in dataset_jobs
    )
    pbar = tqdm(total=n_training_jobs, desc=f"train {config_path.stem}", unit="job")

    for ds_config, folds, classifier_models in dataset_jobs:
        dataset_id = ds_config.dataset_id
        classifier_runs = []

        # Complete every classifier lag sweep before choosing the shared worst group.
        for fold in folds:
            lags = override_lags or ds_config.lags
            for model in classifier_models:
                pbar.set_postfix(dataset=dataset_id, fold=fold, model=model)
                ckpt = train_one_model(
                    ds_config,
                    model_name=model,
                    fold=fold,
                    lags=lags,
                    training=training,
                    output_dir=results_dir / "checkpoints",
                    seed=seed + fold,
                    reuse_checkpoint=not args.retrain,
                )
                pbar.update()
                if not ckpt:
                    continue
                rows = evaluate_classifier_lags(
                    ds_config,
                    ckpt,
                    model,
                    fold,
                    lags,
                    training,
                    eval_batch_size,
                    group_names=[fixed_selected_group]
                    if args.selected_group_only and fixed_selected_group else None,
                )
                eval_rows.setdefault((dataset_id, model), []).append(rows)
                classifier_runs.append((fold, model, ckpt, lags))

        current_rows = [
            frame
            for (row_dataset, _), frames in eval_rows.items()
            if row_dataset == dataset_id
            for frame in frames
        ]
        if fixed_selected_group:
            selected_group = fixed_selected_group
        elif current_rows:
            selected_group = select_syncnet_group_from_frames(current_rows, dataset_id)
        else:
            selected_group = select_syncnet_group(results_dir, dataset_id)
        selected_groups[dataset_id] = selected_group

        correction_lags = (
            [int(lag) for lag in comma_list(args.correction_lags)]
            if args.correction_lags
            else None
        )
        if evaluate_correction:
            for fold, model, ckpt, lags in classifier_runs:
                corrections = evaluate_classifier_corrections(
                    ds_config,
                    ckpt,
                    model,
                    fold,
                    lags,
                    training,
                    eval_batch_size,
                    correction_cache,
                    seed + fold,
                    group_names=[selected_group],
                    imposed_lags=correction_lags,
                )
                correction_rows.setdefault((dataset_id, model), []).append(corrections)

        if "syncnet" in models:
            for fold in folds:
                lags = override_lags or ds_config.lags
                pbar.set_postfix(
                    dataset=dataset_id,
                    fold=fold,
                    model=f"syncnet:{selected_group}",
                )
                ckpt = train_one_model(
                    ds_config,
                    model_name="syncnet",
                    fold=fold,
                    lags=lags,
                    training=training,
                    output_dir=results_dir / "checkpoints",
                    seed=seed + fold,
                    group_name=selected_group,
                    reuse_checkpoint=not args.retrain,
                )
                pbar.update()
                if ckpt:
                    syncnet_rows.setdefault((dataset_id, "syncnet"), []).append(
                        evaluate_syncnet_lags(
                            ds_config,
                            ckpt,
                            fold,
                            selected_group,
                            lags,
                            training,
                            eval_batch_size,
                        )
                    )
    pbar.close()

    results_dir.mkdir(parents=True, exist_ok=True)
    for (dataset_id, model), rows in eval_rows.items():
        pd.concat(rows, ignore_index=True).to_csv(results_dir / f"{dataset_id}_{model}_lag_accuracy.csv", index=False)
    for (dataset_id, model), rows in syncnet_rows.items():
        pd.concat(rows, ignore_index=True).to_csv(results_dir / f"{dataset_id}_{model}_lag_distance.csv", index=False)
    for (dataset_id, model), rows in correction_rows.items():
        pd.concat(rows, ignore_index=True).to_csv(
            results_dir / f"{dataset_id}_{model}_unconditional_correction.csv",
            index=False,
        )
    for dataset_id, selected_group in selected_groups.items():
        pd.DataFrame([{"dataset": dataset_id, "group": selected_group}]).to_csv(
            results_dir / f"{dataset_id}_selected_group.csv", index=False
        )
    for dataset_id in selected_groups:
        curves = []
        for (row_dataset, fold, group_name), correction in correction_cache.items():
            if row_dataset != dataset_id:
                continue
            curve = correction["curve"].copy()
            curve.insert(0, "dataset", dataset_id)
            curve.insert(1, "fold", fold)
            curve.insert(2, "metric", "unconditional")
            curves.append(curve)
        if curves:
            pd.concat(curves, ignore_index=True).to_csv(
                results_dir / f"{dataset_id}_correction_lag_curves.csv", index=False
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--models", help="comma-separated model names")
    parser.add_argument("--folds", help="comma-separated folds or 'all'")
    parser.add_argument(
        "--fixed-groups-from", metavar="RESULTS_DIR",
        help="use each dataset's selected group from an earlier result directory",
    )
    parser.add_argument("--results-dir", default="results/original")
    parser.add_argument("--data-root", help="directory containing local dataset arrays")
    parser.add_argument("--correction-lags", help="comma-separated imposed lags for correction evaluation")
    parser.add_argument("--lags", help="comma-separated training and evaluation lag override")
    parser.add_argument("--skip-corrections", action="store_true")
    parser.add_argument(
        "--selected-group-only",
        action="store_true",
        help="evaluate classifiers only on the fixed selected group",
    )
    parser.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    args = parser.parse_args()
    for path in args.configs:
        run_config(Path(path), args)


if __name__ == "__main__":
    main()

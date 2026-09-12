#!/usr/bin/env python
"""Create every figure and table reported in the paper and appendix.

This is the only plotting/analysis entry point.  It post-processes the original
deployment experiment, the SyncNet baseline comparison, and the whole-dataset
audit.  It never trains a model or recomputes an MDL curve.
"""
import argparse
import os
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch

RESULTS = Path("results/original")
OUTPUT = Path("paper/generated/original")
CONFIGS = Path("configs")
MODEL_ORDER = ["cnn", "resnet", "late_fusion", "magnitude_cnn"]
REAL_DATASETS = [
    "fordchallenge", "opportunity", "pamap2",
    "uciactivity", "skoda", "crowdsourced",
    "dreamera", "dreamerv", "stew",
]
SYNTHETIC_DATASETS = ["syncsine_sensitive", "syncsine_robust"]
# Taller than the default two-panel canvas so the enlarged legend and axis
# labels do not eat into the plotting area.  Shared by the accuracy and metric
# synthetic figures so the two line up when placed above one another.
SYNTHETIC_FIGSIZE = (7.0, 3.4)
# Common left/right axes margins for those two figures.  Wide enough for the
# accuracy grid's three-digit tick labels, which need more room than the
# metric grid's two-digit kbit labels.
SYNTHETIC_XMARGINS = (0.125, 0.954)


def set_theme():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.2,
        "lines.markersize": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


@contextmanager
def larger_fonts(scale=1.8):
    """Enlarge axis and legend text for the two-panel synthetic figures.

    They are printed at the same width as the 3x3 real-dataset grids but carry
    a ninth of the panels, so the shared theme leaves their labels much smaller
    than they need to be relative to the body text.
    """
    keys = (
        "font.size", "axes.titlesize", "axes.labelsize",
        "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
    )
    with plt.rc_context({key: plt.rcParams[key] * scale for key in keys}):
        yield


def display_name(value: str) -> str:
    names = {
        "syncsine_sensitive": "SyncSine-Sensitive",
        "syncsine_robust": "SyncSine-Robust",
        "crowdsourced": "CrowdSourced",
        "dreamera": "DREAMERA",
        "dreamerv": "DREAMERV",
        "fordchallenge": "FordChallenge",
        "opportunity": "Opportunity",
        "pamap2": "PAMAP2",
        "skoda": "Skoda",
        "stew": "STEW",
        "uciactivity": "UCIActivity",
        "cnn": "CNN",
        "resnet": "ResNet",
        "late_fusion": "Late fusion",
        "magnitude_cnn": "Magnitude CNN",
    }
    return names.get(str(value), str(value).replace("_", " "))


def latex_name(value: str) -> str:
    return display_name(value).replace("_", "\\_")


def read_csv_glob(directory: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {directory / pattern}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def read_lag_accuracy(directory: Path) -> pd.DataFrame:
    return read_csv_glob(directory, "*_lag_accuracy.csv")


def read_corrections(directory: Path) -> pd.DataFrame:
    return read_csv_glob(directory, "*_unconditional_correction.csv")


def selected_groups():
    frames = []
    for path in sorted(RESULTS.glob("*_selected_group.csv")):
        frame = pd.read_csv(path)
        if {"dataset", "group"}.issubset(frame.columns):
            frames.append(frame[["dataset", "group"]])
    if not frames:
        raise ValueError(f"{RESULTS}: no selected-group CSVs found")
    rows = pd.concat(frames, ignore_index=True).drop_duplicates("dataset")
    rows.to_csv(OUTPUT / "selected_groups.csv", index=False)
    return dict(zip(rows["dataset"], rows["group"]))


def panel_label(ax, index, dataset):
    ax.text(
        0.03,
        0.96,
        f"({chr(97 + index)}) {display_name(dataset)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )


def panel_caption(ax, index, dataset, xlabel):
    """Put the panel subcaption below the axis label, LaTeX subfigure style.
    """
    ax.set_xlabel(f"{xlabel}\n ", labelpad=2, linespacing=1.5)
    ax.annotate(
        f"({chr(97 + index)}) {display_name(dataset)}",
        xy=(0.5, 0.0),
        xycoords=ax.xaxis.label,
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.labelsize"] + 1,
        fontweight="bold",
    )


def save_grid(fig, path, *, legend=None, legend_frac=0.045, xmargins=None):
    """Save a figure, reserving ``legend_frac`` of the height for the legend.

    The default reproduces the layout used by the full-size grids; enlarged
    figures need a bigger band because the legend grows with the font while the
    canvas does not.  ``xmargins`` overrides the horizontal margins that
    tight_layout derives from the tick labels, so that figures meant to be
    stacked share one left and right edge instead of drifting apart by a few
    millimetres whenever their tick labels differ in width.
    """
    if legend:
        handles, labels, ncols = legend
        fig.legend(handles, labels, loc="upper center", ncols=ncols, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 1.0 - legend_frac))
    else:
        fig.tight_layout()
    if xmargins is not None:
        fig.subplots_adjust(left=xmargins[0], right=xmargins[1])
    fig.savefig(path)
    plt.close(fig)


def grid_axes(datasets, figsize=None):
    if figsize is not None:
        fig, axes = plt.subplots(1, len(datasets), figsize=figsize)
    elif len(datasets) == 9:
        fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    return fig, np.asarray(axes).reshape(-1)


def mean_band(
    ax, frame, x_col, y_col, *, label, color=None, linestyle="-", uncertainty=True,
):
    values = frame.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index().sort_values(x_col)
    marker = None if len(values) > 25 else "o"
    line, = ax.plot(
        values[x_col], values["mean"], label=label, color=color,
        linestyle=linestyle, marker=marker,
        markevery=None if marker is None else max(1, len(values) // 10),
    )
    sd = values["std"].fillna(0.0)
    if uncertainty and (sd > 0).any():
        ax.fill_between(
            values[x_col], values["mean"] - sd, values["mean"] + sd,
            color=line.get_color(), alpha=0.10, linewidth=0,
        )
    return line


def accuracy_grid(
    acc, groups, datasets, path, *, uncertainty=True, figsize=None,
    legend_frac=0.045, xmargins=None,
):
    fig, axes = grid_axes(datasets, figsize=figsize)
    real_grid = len(datasets) == 9
    handles = []
    for index, (ax, dataset) in enumerate(zip(axes, datasets)):
        rows = acc[(acc["dataset"] == dataset) & (acc["group"] == groups[dataset])].copy()
        rows["accuracy_pct"] = 100.0 * rows["accuracy"]
        for model in MODEL_ORDER:
            line = mean_band(
                ax, rows[rows["model"] == model], "lag", "accuracy_pct",
                label=display_name(model), uncertainty=uncertainty,
            )
            if index == 0:
                handles.append(line)
        if real_grid:
            ax.set_title(
                f"({chr(97 + index)}) {display_name(dataset)}",
                pad=4,
            )
            if index >= 6:
                ax.set_xlabel("Sensor delay $\\tau$ [samples]", labelpad=2)
        else:
            panel_caption(ax, index, dataset, "Sensor delay $\\tau$ [samples]")
        if index == 0 or (real_grid and index % 3 == 0):
            ax.set_ylabel("Test accuracy [%]", labelpad=2)
    if not real_grid:
        selected = pd.concat([
            acc[(acc["dataset"] == dataset) & (acc["group"] == groups[dataset])]
            for dataset in datasets
        ])
        lower = 5.0 * np.floor(selected["accuracy"].min() * 100.0 / 5.0)
        for ax in axes:
            ax.set_ylim(lower, 100.0)
    save_grid(
        fig, path,
        legend=(handles, [display_name(model) for model in MODEL_ORDER], 4),
        legend_frac=legend_frac, xmargins=xmargins,
    )


def metric_grid(
    metric, groups, datasets, path, *, shape=None, figsize=None, xmargins=None,
):
    if shape is None:
        fig, axes = grid_axes(datasets, figsize=figsize)
    else:
        fig, axes = plt.subplots(*shape, figsize=(7.0, 5.0))
        axes = np.asarray(axes).reshape(-1)
    real_grid = len(datasets) == 9
    for index, (ax, dataset) in enumerate(zip(axes, datasets)):
        group = groups[dataset]
        rows = metric[
            (metric["dataset"] == dataset) & (metric["group_name"] == group)
        ].copy()
        # Plot in kbit: keeps the tick labels short enough that the panels line
        # up with the accuracy grids, and matches the unit the tables already
        # report Q_G in.
        rows["alignment_cost_kbit"] = rows["alignment_cost_bits"] / 1000.0
        mean_band(ax, rows, "tau", "alignment_cost_kbit", label=group)
        if real_grid or shape is not None:
            ax.set_title(
                f"({chr(97 + index)}) {display_name(dataset)}",
                pad=4,
            )
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ncols = 3 if real_grid else (shape[1] if shape is not None else 2)
        last_row_start = len(datasets) - ncols
        if index >= last_row_start:
            ax.set_xlabel("Sensor delay $\\tau$ [samples]", labelpad=2)
        if index % ncols == 0:
            ax.set_ylabel("Alignment cost $Q_G$ [kbit]", labelpad=2)
    if not real_grid and shape is None:
        selected = metric[metric["dataset"].isin(datasets)]
        upper = 0.0
        for dataset in datasets:
            group = groups[dataset]
            rows = selected[
                (selected["dataset"] == dataset)
                & (selected["group_name"] == group)
            ]
            band = rows.groupby("tau")["alignment_cost_bits"].agg(["mean", "std"])
            upper = max(upper, float((band["mean"] + band["std"].fillna(0.0)).max()))
        upper = 10.0 * np.ceil(upper / 1000.0 / 10.0)
        for ax in axes:
            ax.set_ylim(0.0, upper)
    save_grid(fig, path, xmargins=xmargins)


def synthetic_example_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    class_colors = {0: "#fecaca", 1: "#dcfce7"}
    samples_per_window = 200
    duration = 2.0

    # Sensitive schematic: two in-phase channels.  A stretch of time is class 1
    # exactly while x1 + x2 exceeds the threshold, so the shading follows the
    # crossings of the sum instead of a fixed window grid.  Amplitudes and
    # offsets are chosen for two properties: neither channel reaches the
    # threshold on its own -- only their sum does, so the panel cannot be read
    # as a per-channel rule -- and the class-1 share of the trace is 30%, close
    # to the dataset's 31/69 balance.  For x1 + x2 = S sin(2 pi f t) + O that
    # share is 1/2 - arcsin((1 - tau) / S) / pi, which fixes S = 1.50, O = 0.12.
    threshold = 1.0
    sensitive_time = np.linspace(0.0, duration, 2000, endpoint=False)
    shared = np.sin(2.0 * np.pi * sensitive_time)
    sensitive_0 = 0.68 * shared + 0.29
    sensitive_1 = 0.82 * shared - 0.17

    # Robust schematic: channel 1 is one continuous chirp whose instantaneous
    # frequency moves smoothly across the class threshold. Channel 2 is a
    # continuous nuisance oscillation. Labels follow the mean frequency of
    # channel 1 in each window.
    n_windows = 8
    dt = duration / (n_windows * samples_per_window)
    time = np.arange(n_windows * samples_per_window) * dt
    frequency_threshold = 5.0
    instantaneous_frequency = frequency_threshold + 3.0 * np.sin(
        4.0 * np.pi * time / duration
    )
    phase = 2.0 * np.pi * np.cumsum(instantaneous_frequency) * dt
    robust_0 = np.sin(phase)
    robust_1 = 0.85 * np.sin(2.0 * np.pi * 3.2 * time + 0.7)
    robust_labels = []
    for window in range(n_windows):
        start = window * samples_per_window
        stop = start + samples_per_window
        robust_labels.append(
            int(instantaneous_frequency[start:stop].mean() > frequency_threshold)
        )

    # Sensitive panel: shade the runs delimited by the crossings of x1 + x2.
    above = (sensitive_0 + sensitive_1) > threshold
    cuts = np.flatnonzero(np.diff(above.astype(int))) + 1
    bounds = np.concatenate(([0], cuts, [above.size]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        start = sensitive_time[lo] if lo == 0 else sensitive_time[lo - 1:lo + 1].mean()
        stop = (
            sensitive_time[-1] if hi == above.size
            else sensitive_time[hi - 1:hi + 1].mean()
        )
        axes[0].axvspan(
            start, stop, color=class_colors[int(above[lo])], alpha=0.45,
            linewidth=0, zorder=0,
        )

    # Robust panel: labels are constant over each fixed window.
    for window, label in enumerate(robust_labels):
        axes[1].axvspan(
            window * samples_per_window * dt,
            (window + 1) * samples_per_window * dt,
            color=class_colors[int(label)], alpha=0.45, linewidth=0, zorder=0,
        )

    for ax, panel_time, channel_0, channel_1 in [
        (axes[0], sensitive_time, sensitive_0, sensitive_1),
        (axes[1], time, robust_0, robust_1),
    ]:
        ax.plot(panel_time, channel_0, label="Channel 1")
        ax.plot(panel_time, channel_1, label="Channel 2")
        ax.set_xlim(panel_time[0], panel_time[-1])
        ax.set_xlabel("Time [s]", labelpad=2)
    axes[0].set_ylabel("Signal value", labelpad=2)

    handles = [
        axes[0].lines[0],
        axes[0].lines[1],
        Patch(facecolor=class_colors[0], edgecolor="none", label="Class $y=0$"),
        Patch(facecolor=class_colors[1], edgecolor="none", label="Class $y=1$"),
    ]
    labels = [handle.get_label() for handle in handles]
    save_grid(
        fig,
        OUTPUT / "paper_synthetic_examples.pdf",
        legend=(handles, labels, 4),
    )


def correction_grid(corrections, groups, datasets, path, *, uncertainty=True):
    fig, axes = grid_axes(datasets)
    real_grid = len(datasets) == 9
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = []
    labels = []
    for index, (ax, dataset) in enumerate(zip(axes, datasets)):
        rows = corrections[
            (corrections["dataset"] == dataset)
            & (corrections["group"] == groups[dataset])
        ].copy()
        rows["lagged_pct"] = 100.0 * rows["lagged_accuracy"]
        rows["corrected_pct"] = 100.0 * rows["corrected_accuracy"]
        for model_index, model in enumerate(MODEL_ORDER):
            model_rows = rows[rows["model"] == model]
            for column, style, suffix in [
                ("lagged_pct", "-", "delayed"),
                ("corrected_pct", "--", "corrected"),
            ]:
                line = mean_band(
                    ax, model_rows, "imposed_lag", column,
                    label=f"{display_name(model)} {suffix}",
                    color=colors[model_index], linestyle=style,
                    uncertainty=uncertainty,
                )
                if index == 0:
                    handles.append(line)
                    labels.append(f"{display_name(model)} {suffix}")
        if real_grid:
            ax.set_title(
                f"({chr(97 + index)}) {display_name(dataset)}",
                pad=4,
            )
        else:
            panel_label(ax, index, dataset)
        ax.set_xlabel("Imposed delay $\\tau$ [samples]", labelpad=2)
        ax.set_ylabel("Test accuracy [%]", labelpad=2)
    save_grid(fig, path, legend=(handles, labels, 4))


def accuracy_correction_table(acc, corrections, groups):
    lines = [
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "Dataset & Group & CNN & ResNet & Late fusion & Magnitude \\\\",
        "\\midrule",
    ]
    for index, dataset in enumerate(REAL_DATASETS + SYNTHETIC_DATASETS):
        if index == len(REAL_DATASETS):
            lines.append("\\midrule")
        cells = []
        for model in MODEL_ORDER:
            model_rows = acc[
                (acc["dataset"] == dataset)
                & (acc["group"] == groups[dataset])
                & (acc["model"] == model)
            ].copy()
            baseline = model_rows[model_rows["lag"] == 0][
                ["fold", "accuracy"]
            ].rename(columns={"accuracy": "baseline_accuracy"})
            rows = model_rows[model_rows["lag"] != 0].merge(
                baseline, on="fold", validate="many_to_one"
            )
            worst = rows.loc[
                rows.groupby("fold")["accuracy"].idxmin(),
                ["fold", "lag", "baseline_accuracy", "accuracy"],
            ]
            corrected = corrections[
                (corrections["dataset"] == dataset)
                & (corrections["group"] == groups[dataset])
                & (corrections["model"] == model)
            ][["fold", "imposed_lag", "corrected_accuracy"]]
            joined = worst.merge(
                corrected,
                left_on=["fold", "lag"],
                right_on=["fold", "imposed_lag"],
                validate="one_to_one",
            )
            nominal = 100.0 * joined["baseline_accuracy"].mean()
            delayed = 100.0 * joined["accuracy"].mean()
            recovered = 100.0 * joined["corrected_accuracy"].mean()
            loss = nominal - delayed
            if abs(loss) < 0.05:
                loss = 0.0
            loss_text = "0.0" if loss == 0.0 else f"{-loss:.1f}"
            cells.append(
                f"{nominal:.1f}$\\rightarrow${delayed:.1f}$\\rightarrow${recovered:.1f} "
                f"[${loss_text}$]"
            )
        lines.append(
            f"{latex_name(dataset)} & {latex_name(groups[dataset])} & "
            + " & ".join(cells) + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (OUTPUT / "paper_accuracy_correction_table.tex").write_text("\n".join(lines) + "\n")


def nominal_lag_table(metric, groups):
    selected = pd.concat([
        metric[
            (metric["dataset"] == dataset)
            & (metric["group_name"] == group)
        ]
        for dataset, group in groups.items()
    ])
    per_fold = (
        selected.groupby(["dataset", "fold", "group_name"], as_index=False)
        .agg(
            nominal_lag=("best_lag", "first"),
            cost_at_zero_bits=("alignment_cost_bits", lambda values: float(
                values.loc[selected.loc[values.index, "tau"] == 0].iloc[0]
            )),
        )
    )
    nominal = (
        per_fold.groupby(["dataset", "group_name"], as_index=False)
        .agg(
            nominal_lag=("nominal_lag", "median"),
            minimum_nominal_lag=("nominal_lag", "min"),
            maximum_nominal_lag=("nominal_lag", "max"),
            cost_at_zero_bits=("cost_at_zero_bits", "mean"),
            zero_best_fraction=("nominal_lag", lambda values: float((values == 0).mean())),
        )
    )
    nominal.to_csv(OUTPUT / "nominal_lags.csv", index=False)
    real = [
        "crowdsourced", "dreamera", "dreamerv", "fordchallenge", "opportunity",
        "pamap2", "skoda", "stew", "uciactivity",
    ]
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Group & $\\tau_G^\\star$ & Fold range & $Q_G(0)$ (kbit) & Zero best \\\\",
        "\\midrule",
    ]
    for dataset in real:
        group = groups[dataset]
        row = nominal[
            (nominal["dataset"] == dataset) & (nominal["group_name"] == group)
        ].iloc[0]
        lines.append(
            f"{latex_name(dataset)} & {latex_name(group)} & {row.nominal_lag:.0f} & "
            f"{row.minimum_nominal_lag:.0f}--{row.maximum_nominal_lag:.0f} & "
            f"{row.cost_at_zero_bits / 1000.0:.1f} & "
            f"{int(round(5 * row.zero_best_fraction))}/5 \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (OUTPUT / "paper_nominal_lags.tex").write_text("\n".join(lines) + "\n")


def selected_metric(path: Path, groups: dict[str, str]) -> pd.DataFrame:
    metric = pd.read_csv(path)
    wanted = pd.DataFrame([
        {"dataset": dataset, "group_name": group}
        for dataset, group in groups.items()
    ])
    return metric.merge(wanted, on=["dataset", "group_name"], how="inner")


def argmin_lag(frame: pd.DataFrame, lag_column: str, score_column: str) -> int:
    minimum = frame[score_column].min()
    return int(frame.loc[frame[score_column] == minimum, lag_column].min())


def syncnet_comparison(original: Path, groups: dict[str, str]) -> pd.DataFrame:
    """Compare MDL and SyncNet on the common, baseline-reachable lag grid."""
    train_metric = selected_metric(
        original / "unconditional" / "lag_curves.csv", groups,
    )
    train_syncnet = pd.read_csv(original / "syncnet_train_lag_distance.csv")
    rows = []
    for dataset, group in groups.items():
        mdl_test = pd.read_csv(original / f"{dataset}_correction_lag_curves.csv")
        mdl_test = mdl_test[
            (mdl_test["metric"] == "unconditional")
            & (mdl_test["group_name"] == group)
        ]
        sync_test = pd.read_csv(original / f"{dataset}_syncnet_lag_distance.csv")
        sync_test = sync_test[sync_test["group"] == group]

        for fold in sorted(set(mdl_test["fold"]) & set(sync_test["fold"])):
            mdl_curve = mdl_test[mdl_test["fold"] == fold][
                ["tau", "alignment_cost_bits"]
            ]
            sync_curve = sync_test[sync_test["fold"] == fold][
                ["lag", "mean_distance"]
            ]
            common = sorted(set(mdl_curve["tau"]) & set(sync_curve["lag"]))
            mdl_curve = mdl_curve[mdl_curve["tau"].isin(common)]
            sync_curve = sync_curve[sync_curve["lag"].isin(common)]
            mdl_train = train_metric[
                (train_metric["dataset"] == dataset)
                & (train_metric["fold"] == fold)
            ]
            sync_train = train_syncnet[
                (train_syncnet["dataset"] == dataset)
                & (train_syncnet["fold"] == fold)
                & (train_syncnet["group"] == group)
            ]
            mdl_baseline = argmin_lag(mdl_train, "tau", "alignment_cost_bits")
            sync_baseline = argmin_lag(sync_train, "lag", "mean_distance")
            for imposed in common:
                mdl_selected = argmin_lag(
                    mdl_curve[mdl_curve["tau"] <= imposed],
                    "tau", "alignment_cost_bits",
                )
                sync_selected = argmin_lag(
                    sync_curve[sync_curve["lag"] <= imposed],
                    "lag", "mean_distance",
                )
                rows.append({
                    "dataset": dataset,
                    "fold": int(fold),
                    "group": group,
                    "imposed_lag": int(imposed),
                    "mdl_baseline_lag": mdl_baseline,
                    "syncnet_baseline_lag": sync_baseline,
                    "mdl_selected_lag": mdl_selected,
                    "syncnet_selected_lag": sync_selected,
                    "mdl_error": abs(mdl_selected - mdl_baseline),
                    "syncnet_error": abs(sync_selected - sync_baseline),
                    "common_grid_size": len(common),
                    "jointly_reachable": imposed >= max(mdl_baseline, sync_baseline),
                })
    return pd.DataFrame(rows)


def dataset_geometry(dataset: str) -> dict[str, int]:
    """Window geometry of one dataset, taken from its experiment config.

    ``window_length`` is the original length of a source window, before the
    largest imposed lag is reserved, and ``reduced_window_length`` is what the
    models actually see once the lag has been applied.  The SyncSine windows are shorter than
    their lag range, so no reduced window exists there and the lag range itself
    is used as the normaliser instead.
    """
    path = CONFIGS / f"{dataset}.yml"
    config = yaml.safe_load(path.read_text())
    if "window_length" not in config:
        raise ValueError(f"{path}: missing required field 'window_length'")
    window_length = int(config["window_length"])
    max_lag = max(int(lag) for lag in config["lags"])
    reduced = window_length - max_lag
    return {
        "window_length": window_length,
        "max_lag": max_lag,
        "reduced_window_length": reduced,
        "reduced_window_normaliser": reduced if reduced > 0 else max_lag,
    }


def write_error_table(
    summary: pd.DataFrame,
    path: Path,
    mdl_column: str,
    syncnet_column: str,
    precision: int,
):
    """Write one MDL-versus-SyncNet lag error table over the fixed order."""
    lines = [
        "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
        "Dataset & MDL & SyncNet & Cases \\\\", "\\midrule",
    ]
    for dataset in REAL_DATASETS + SYNTHETIC_DATASETS:
        row = summary[summary["dataset"] == dataset].iloc[0]
        lines.append(
            f"{latex_name(dataset)} & {row[mdl_column]:.{precision}f} & "
            f"{row[syncnet_column]:.{precision}f} & {int(row.cases)} \\\\"
        )
        if dataset == REAL_DATASETS[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_syncnet_comparison(
    rows: pd.DataFrame,
    accuracy: pd.DataFrame,
    output: Path,
):
    """Write auditable rows plus the compact appendix tables."""
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "syncnet_comparison_rows.csv", index=False)
    reachable = rows[rows["jointly_reachable"]].copy()
    error = (
        reachable.groupby("dataset", as_index=False)
        .agg(
            mdl_error=("mdl_error", "mean"),
            syncnet_error=("syncnet_error", "mean"),
            cases=("imposed_lag", "size"),
        )
    )

    accuracy_lookup = accuracy.set_index(
        ["dataset", "fold", "model", "group", "lag"]
    )["accuracy"]
    accuracy_rows = []
    for row in reachable.itertuples():
        for model in MODEL_ORDER:
            prefix = (row.dataset, row.fold, model, row.group)
            keys = {
                "nominal": (*prefix, 0),
                "lagged": (*prefix, row.imposed_lag),
                "mdl": (*prefix, row.mdl_selected_lag),
                "syncnet": (*prefix, row.syncnet_selected_lag),
            }
            if all(key in accuracy_lookup.index for key in keys.values()):
                accuracy_rows.append({
                    "dataset": row.dataset,
                    **{name: float(accuracy_lookup.loc[key]) for name, key in keys.items()},
                })
    accuracy_summary = pd.DataFrame(accuracy_rows).groupby(
        "dataset", as_index=False,
    ).mean(numeric_only=True)
    summary = error.merge(accuracy_summary, on="dataset", validate="one_to_one")
    geometry = pd.DataFrame([
        {"dataset": dataset, **dataset_geometry(dataset)}
        for dataset in summary["dataset"]
    ])
    summary = summary.merge(geometry, on="dataset", validate="one_to_one")
    # Report the mean absolute lag error as a score: the fraction of the
    # normaliser that the selector did *not* waste, in percent.
    for model in ("mdl", "syncnet"):
        summary[f"{model}_score_reduced_window"] = 100.0 * (
            1.0 - summary[f"{model}_error"] / summary["reduced_window_normaliser"]
        )
        summary[f"{model}_score_max_lag"] = 100.0 * (
            1.0 - summary[f"{model}_error"] / summary["max_lag"]
        )
    summary.to_csv(output / "syncnet_comparison_summary.csv", index=False)

    write_error_table(
        summary, output / "paper_syncnet_error_table.tex",
        "mdl_error", "syncnet_error", 2,
    )
    write_error_table(
        summary,
        output / "paper_syncnet_error_table_norm_by_reduced_window_length.tex",
        "mdl_score_reduced_window", "syncnet_score_reduced_window", 2,
    )
    write_error_table(
        summary, output / "paper_syncnet_error_table_norm_by_max_lag.tex",
        "mdl_score_max_lag", "syncnet_score_max_lag", 2,
    )

    order = REAL_DATASETS + SYNTHETIC_DATASETS
    lines = [
        "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
        "Dataset & Nominal & Lagged & MDL & SyncNet \\\\", "\\midrule",
    ]
    for dataset in order:
        row = summary[summary["dataset"] == dataset].iloc[0]
        lines.append(
            f"{latex_name(dataset)} & {100 * row.nominal:.1f} & "
            f"{100 * row.lagged:.1f} & {100 * row.mdl:.1f} & "
            f"{100 * row.syncnet:.1f} \\\\"
        )
        if dataset == REAL_DATASETS[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "paper_syncnet_accuracy_table.tex").write_text("\n".join(lines) + "\n")


def lag_zero_accuracy(results: Path) -> pd.DataFrame:
    accuracy = read_lag_accuracy(results)
    zero = accuracy[accuracy["lag"] == 0]
    spread = zero.groupby(["dataset", "fold", "model"])["accuracy"].agg(
        lambda values: float(values.max() - values.min())
    )
    if (spread > 1e-12).any():
        raise ValueError("lag-zero accuracy unexpectedly depends on the sensor group")
    return zero.groupby(
        ["dataset", "fold", "model"], as_index=False,
    )["accuracy"].first()


def global_audit_rows(
    audit: Path,
    original: Path,
    corrected: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    audit_curves = pd.read_csv(audit / "unconditional" / "lag_curves.csv")
    original_curves = pd.read_csv(original / "unconditional" / "lag_curves.csv")
    alignment_rows = []
    curves = []
    for item in manifest.itertuples():
        selected = audit_curves[
            (audit_curves["dataset"] == item.dataset)
            & (audit_curves["group_name"] == item.group)
        ].copy()
        curves.append(selected)
        fold_lags = original_curves[
            (original_curves["dataset"] == item.dataset)
            & (original_curves["group_name"] == item.group)
        ].groupby("fold")["best_lag"].first()
        alignment_rows.append({
            "dataset": item.dataset,
            "group": item.group,
            "global_lag": int(item.global_lag),
            "fold_lag_min": int(fold_lags.min()),
            "fold_lag_max": int(fold_lags.max()),
            "cost_at_zero_bits": float(item.cost_at_zero_bits),
            "at_boundary": bool(item.at_boundary),
        })
    original_accuracy = lag_zero_accuracy(original).rename(
        columns={"accuracy": "original_accuracy"},
    )
    corrected_accuracy = lag_zero_accuracy(corrected).rename(
        columns={"accuracy": "corrected_accuracy"},
    )
    accuracy_rows = original_accuracy.merge(
        corrected_accuracy,
        on=["dataset", "fold", "model"],
        validate="one_to_one",
    )
    accuracy_rows["accuracy_change"] = (
        accuracy_rows["corrected_accuracy"] - accuracy_rows["original_accuracy"]
    )
    return (
        pd.DataFrame(alignment_rows),
        pd.concat(curves, ignore_index=True),
        accuracy_rows,
    )


def write_global_audit(
    alignment: pd.DataFrame,
    curves: pd.DataFrame,
    accuracy: pd.DataFrame,
    output: Path,
):
    output.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(output / "global_alignment_summary.csv", index=False)
    accuracy.to_csv(output / "global_accuracy_rows.csv", index=False)
    summary = accuracy.groupby(["dataset", "model"], as_index=False).agg(
        original_accuracy=("original_accuracy", "mean"),
        corrected_accuracy=("corrected_accuracy", "mean"),
        accuracy_change=("accuracy_change", "mean"),
    )
    summary.to_csv(output / "global_accuracy_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{@{}llrrrc@{}}", "\\toprule",
        "Dataset & Group & Global lag & Fold range & $Q_G(0)$ [kbit] & Boundary \\\\",
        "\\midrule",
    ]
    for dataset in REAL_DATASETS:
        row = alignment[alignment["dataset"] == dataset].iloc[0]
        lines.append(
            f"{latex_name(dataset)} & {latex_name(row.group)} & {row.global_lag} & "
            f"{row.fold_lag_min}--{row.fold_lag_max} & "
            f"{row.cost_at_zero_bits / 1000:.1f} & "
            f"{'yes' if row.at_boundary else 'no'} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "paper_global_alignment_table.tex").write_text("\n".join(lines) + "\n")

    lines = [
        "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
        "Dataset & CNN & ResNet & Late fusion & Magnitude \\\\", "\\midrule",
    ]
    for dataset in REAL_DATASETS:
        cells = []
        for model in MODEL_ORDER:
            row = summary[
                (summary["dataset"] == dataset) & (summary["model"] == model)
            ].iloc[0]
            cells.append(
                f"{100 * row.original_accuracy:.1f}$\\rightarrow$"
                f"{100 * row.corrected_accuracy:.1f} "
                f"[{100 * row.accuracy_change:+.1f}]"
            )
        lines.append(f"{latex_name(dataset)} & " + " & ".join(cells) + " \\\\ ")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "paper_global_accuracy_table.tex").write_text("\n".join(lines) + "\n")

    groups = dict(zip(alignment["dataset"], alignment["group"]))
    metric_grid(curves, groups, REAL_DATASETS, output / "paper_global_alignment_real.pdf")
    metric_grid(
        curves, groups, ["fordchallenge", "pamap2", "skoda", "stew"],
        output / "paper_global_alignment_highlights.pdf", shape=(2, 2),
    )


def write_conditional_comparison(original: Path, groups: dict[str, str], output: Path):
    rows = []
    for variant in ("unconditional", "conditional"):
        summary = pd.read_csv(original / variant / "group_summary.csv")
        wanted = pd.DataFrame([
            {"dataset": dataset, "group_name": group}
            for dataset, group in groups.items()
        ])
        summary = summary.merge(wanted, on=["dataset", "group_name"])
        for (dataset, group), frame in summary.groupby(["dataset", "group_name"]):
            rows.append({
                "variant": variant,
                "dataset": dataset,
                "group": group,
                "best_lag": frame["best_lag"].median(),
                "minimum_lag": frame["best_lag"].min(),
                "maximum_lag": frame["best_lag"].max(),
                "mean_cost_bits": frame["mean_alignment_cost_bits"].mean(),
            })
    comparison = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "conditional_comparison.csv", index=False)

    order = REAL_DATASETS + SYNTHETIC_DATASETS
    lines = [
        "\\begin{tabular}{@{}llrr@{}}", "\\toprule",
        "Dataset & Group & U lag & C lag \\\\", "\\midrule",
    ]
    for dataset in order:
        selected = comparison[comparison["dataset"] == dataset].set_index("variant")
        unconditional = selected.loc["unconditional"]
        conditional = selected.loc["conditional"]
        lines.append(
            f"{latex_name(dataset)} & {latex_name(unconditional.group)} & "
            f"{unconditional.best_lag:.0f} "
            f"[{unconditional.minimum_lag:.0f}--{unconditional.maximum_lag:.0f}] & "
            f"{conditional.best_lag:.0f} "
            f"[{conditional.minimum_lag:.0f}--{conditional.maximum_lag:.0f}] \\\\")
        if dataset == REAL_DATASETS[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "paper_conditional_lags.tex").write_text("\n".join(lines) + "\n")

    lines = [
        "\\begin{tabular}{@{}lrr@{}}", "\\toprule",
        "Dataset & U $A_G$ & C $A_G$ \\\\", "\\midrule",
    ]
    for dataset in order:
        selected = comparison[comparison["dataset"] == dataset].set_index("variant")
        lines.append(
            f"{latex_name(dataset)} & "
            f"{selected.loc['unconditional', 'mean_cost_bits'] / 1000:.1f} & "
            f"{selected.loc['conditional', 'mean_cost_bits'] / 1000:.1f} \\\\")
        if dataset == REAL_DATASETS[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "paper_conditional_costs.tex").write_text("\n".join(lines) + "\n")


def main():
    global RESULTS, OUTPUT, CONFIGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--original-results", default="results/original")
    parser.add_argument("--global-audit-results", default="results/global_audit")
    parser.add_argument("--global-corrected-results", default="results/global_corrected")
    parser.add_argument("--global-manifest", default="data/global_corrected/manifest.csv")
    parser.add_argument("--output-dir", default="paper/generated")
    args = parser.parse_args()

    RESULTS = Path(args.original_results)
    CONFIGS = Path(args.config_dir)
    output_root = Path(args.output_dir)
    OUTPUT = output_root / "original"

    set_theme()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = selected_groups()
    accuracy = read_lag_accuracy(RESULTS)
    metric = pd.read_csv(RESULTS / "unconditional" / "lag_curves.csv")
    corrections = read_corrections(RESULTS)

    accuracy_grid(accuracy, groups, REAL_DATASETS, OUTPUT / "paper_accuracy_real.pdf")
    accuracy_grid(
        accuracy, groups, REAL_DATASETS,
        OUTPUT / "paper_accuracy_real_no_uncertainty.pdf",
        uncertainty=False,
    )
    with larger_fonts():
        accuracy_grid(
            accuracy, groups, SYNTHETIC_DATASETS,
            OUTPUT / "paper_accuracy_synthetic.pdf",
            figsize=SYNTHETIC_FIGSIZE, legend_frac=0.13,
            xmargins=SYNTHETIC_XMARGINS,
        )
    metric_grid(metric, groups, REAL_DATASETS, OUTPUT / "paper_metric_real.pdf")
    metric_grid(
        metric, groups,
        ["fordchallenge", "pamap2", "skoda", "stew"],
        OUTPUT / "paper_metric_highlights.pdf",
        shape=(2, 2),
    )
    with larger_fonts():
        metric_grid(
            metric, groups, SYNTHETIC_DATASETS,
            OUTPUT / "paper_metric_synthetic.pdf", figsize=SYNTHETIC_FIGSIZE,
            xmargins=SYNTHETIC_XMARGINS,
        )
        synthetic_example_figure()
    correction_grid(corrections, groups, REAL_DATASETS, OUTPUT / "paper_correction_real.pdf")
    correction_grid(
        corrections, groups, REAL_DATASETS,
        OUTPUT / "paper_correction_real_no_uncertainty.pdf",
        uncertainty=False,
    )
    correction_grid(corrections, groups, SYNTHETIC_DATASETS, OUTPUT / "paper_correction_synthetic.pdf")
    accuracy_correction_table(accuracy, corrections, groups)
    nominal_lag_table(metric, groups)
    write_conditional_comparison(RESULTS, groups, output_root / "conditional")

    comparison = syncnet_comparison(RESULTS, groups)
    write_syncnet_comparison(comparison, accuracy, output_root / "syncnet")

    alignment, audit_curves, audit_accuracy = global_audit_rows(
        Path(args.global_audit_results),
        RESULTS,
        Path(args.global_corrected_results),
        Path(args.global_manifest),
    )
    write_global_audit(
        alignment, audit_curves, audit_accuracy, output_root / "global_audit",
    )
    print(f"wrote paper figures and tables to {output_root}", flush=True)


if __name__ == "__main__":
    main()

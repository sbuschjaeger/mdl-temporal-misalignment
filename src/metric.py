import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.data import build_dataset, raw_windows_numpy


def mdl_alignment_loss(
    X,
    y=None,
    max_lag=None,
    lags=None,
    groups=None,
    rank=1,
    quant_scale=64.0,
    coef_bits=16,
    eps=1e-9,
    circular=True,
    fixed_window_size=None,
    return_details=False,
):
    """Measure temporal alignment using the paper's raw MDL alignment cost.

    This function implements the construction in the paper's section
    *Alignment Relevance via MDL Compressibility*.  For every candidate sensor
    group ``G`` and lag ``tau``, it compares two descriptions of the shifted
    windows ``Z_G^tau``:

    * the independent code encodes the target group directly; and
    * the shared-representation code reconstructs the target group from an
      anchored rank-``rank`` PCA representation of its unshifted complement,
      then encodes the representation, reconstruction parameters, and
      residuals.

    Their difference is the shared-representation gain ``R_G(tau)``.  The
    returned lag curve reports ``alignment_cost_bits`` as
    ``max_tau R_G(tau) - R_G(tau)``.  This is the paper's ``Q_G(tau)``: it is
    measured in raw bits, is non-negative, and must be *minimized*.  No
    cross-dataset or min--max normalization is applied.

    Labels are optional.  With ``y=None`` all windows are coded jointly and
    the metric is label-free.  Supplying ``y`` activates the class-conditional
    appendix variant: a representation and reconstruction are fitted within
    each class and their codelength gains are summed.  There is intentionally
    no separate mode flag, because the presence of side information fully
    determines the definition.

    Parameters
    ----------
    X : array-like, shape (n_windows, n_channels, n_timepoints)
        Real-valued multichannel windows.  Channel statistics and the fixed
        quantization scale are computed once before evaluating groups/lags.
    y : array-like, shape (n_windows,), optional
        Class labels used as side information.  Omit for the unconditional,
        deployment-compatible metric used throughout the main paper.
    max_lag : int, optional
        Evaluate every integer lag from zero through ``max_lag``.  Ignored
        when the explicit ``lags`` grid is supplied.
    lags : iterable of int, optional
        Configured non-negative lag grid.  Zero is inserted automatically.
    groups : iterable or mapping of channel-index collections, optional
        Candidate target sensor groups.  Defaults to every singleton channel.
    rank : int
        Rank of the PCA representation anchored in the complement.
    quant_scale : float
        Quantization units per channel standard deviation (64 in the paper).
    coef_bits : int
        Fixed codelength assigned to each reconstruction parameter.
    circular : bool
        Use circular shifts.  Experiments set this to ``False`` and crop to a
        common valid region to avoid padding artifacts.
    fixed_window_size : int, optional
        Common cropped length used for every evaluated lag.
    return_details : bool
        Also return per-group summaries, complete lag curves, and quantized
        windows.

    Returns
    -------
    dict or (dict, dict)
        The first dictionary summarizes the group with largest mean non-zero
        alignment cost ``A_G``.  With ``return_details=True``, the second
        dictionary contains ``group_summary`` and ``lag_curves`` data frames.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_windows, n_channels, n_timepoints)")
    if y is not None:
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must contain one label per window")
    _, C, T = X.shape
    fixed_window_size = None if fixed_window_size is None else int(fixed_window_size)
    if fixed_window_size is not None:
        if fixed_window_size <= 0:
            raise ValueError("fixed_window_size must be positive")
        circular = False

    if groups is None:
        groups = [{c} for c in range(C)]
    elif isinstance(groups, dict):
        groups = [set(channels) for channels in groups.values()]
    else:
        groups = [set(channels) for channels in groups]
    groups = [tuple(sorted(group)) for group in groups]

    if lags is None:
        if max_lag is None:
            raise ValueError("Either max_lag or lags must be given.")
        lags = list(range(int(max_lag) + 1))
    else:
        lags = sorted(set(int(lag) for lag in lags) | {0})
    if min(lags) < 0:
        raise ValueError("experiment lags must be non-negative")

    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + eps
    Q = np.rint(quant_scale * (X - mu) / sd).astype(np.int32)

    def entropy_bits(values):
        values = np.asarray(values).ravel()
        if values.size == 0:
            return 0.0
        integers = values.astype(np.int64, copy=False)
        lo = int(integers.min())
        hi = int(integers.max())
        span = hi - lo + 1
        if span <= max(4096, 4 * integers.size):
            counts = np.bincount(integers - lo)
            counts = counts[counts > 0]
        else:
            _, counts = np.unique(integers, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-np.sum(counts * np.log2(probabilities)))

    def temporal_bits(values):
        """Original temporal code: first symbols plus within-window deltas."""
        values = np.asarray(values)
        if values.shape[-1] <= 1:
            return entropy_bits(values)
        return entropy_bits(values[..., :1]) + entropy_bits(np.diff(values, axis=-1))

    def temporal_bits_raw(values):
        """Original residual code: empirical entropy of residual symbols."""
        return entropy_bits(values)

    def pca_representation(context):
        """Return the anchored rank-r representation of the complement."""
        n_samples, n_channels, n_time = context.shape
        if n_channels == 1:
            return context[:, :1, :].astype(float)
        matrix = context.transpose(0, 2, 1).reshape(-1, n_channels).astype(float)
        matrix -= matrix.mean(axis=0, keepdims=True)
        n_components = min(int(rank), n_channels)
        if n_components < 1:
            raise ValueError("rank must be >= 1")
        covariance = matrix.T @ matrix
        _, eigenvectors = np.linalg.eigh(covariance)
        components = eigenvectors[:, -n_components:]
        # PCA signs are arbitrary. Orient each component deterministically so
        # that non-negative reconstruction weights have a stable meaning.
        for component in range(n_components):
            pivot = np.abs(components[:, component]).argmax()
            if components[pivot, component] < 0:
                components[:, component] *= -1
        scores = matrix @ components
        return scores.reshape(n_samples, n_time, n_components).transpose(0, 2, 1)

    def nonnegative_reconstruction(representation_design, target_values):
        """Fit non-negative representation weights and a free intercept."""
        representation_mean = representation_design.mean(axis=0)
        target_mean = target_values.mean(axis=0)
        centered_representation = representation_design - representation_mean
        centered_target = target_values - target_mean
        slopes = np.column_stack([
            nnls(centered_representation, centered_target[:, channel])[0]
            for channel in range(target_values.shape[1])
        ])
        intercept = target_mean - representation_mean @ slopes
        design = np.c_[
            representation_design,
            np.ones((representation_design.shape[0], 1)),
        ]
        weights = np.vstack([slopes, intercept])
        return design, weights

    def slices_for_lag(tau):
        """Keep the complement fixed and delay only the target group."""
        tau = int(tau)
        if tau < 0:
            raise ValueError("experiment lags must be non-negative")
        if fixed_window_size is not None:
            if tau + fixed_window_size > T:
                raise ValueError("Lag too large for fixed overlap window")
            return slice(0, fixed_window_size), slice(tau, tau + fixed_window_size)
        if circular:
            return slice(None), slice(None)
        if tau >= T:
            raise ValueError("Lag too large for cropped shifting")
        return slice(0, T - tau), slice(tau, T)

    quantized_sets = [Q] if y is None else [Q[y == label] for label in np.unique(y)]

    all_curves = []
    group_rows = []
    for group in groups:
        target_channels = list(group)
        context_channels = [channel for channel in range(C) if channel not in group]
        representation_cache = {}

        def representation_model(class_index, context, context_slice):
            # Keep the anchored representation contribution explicit in the
            # shared-representation code described in the paper.
            key = (class_index, context_slice.start, context_slice.stop, context.shape[-1])
            if key not in representation_cache:
                representation = pca_representation(context)
                n_samples, n_components, n_time = representation.shape
                design = representation.transpose(0, 2, 1).reshape(-1, n_components)
                representation_bits = temporal_bits(np.rint(representation).astype(np.int32))
                representation_cache[key] = (
                    design,
                    representation_bits,
                    (n_components + 1) * len(group) * coef_bits,
                    n_samples,
                    n_time,
                )
            return representation_cache[key]

        curve_rows = []
        for tau in lags:
            context_slice, target_slice = slices_for_lag(tau)
            gain_bits = 0.0
            target_bits = 0.0
            shared_target_bits = 0.0
            for class_index, quantized in enumerate(quantized_sets):
                context = quantized[:, context_channels, context_slice]
                target = quantized[:, target_channels, target_slice]
                if circular and tau:
                    target = np.roll(target, shift=-int(tau), axis=2)

                direct_bits = temporal_bits(target)
                representation_design, representation_bits, model_bits, n_samples, n_time = representation_model(
                    class_index, context, context_slice
                )
                values = target.transpose(0, 2, 1).reshape(-1, len(group)).astype(float)
                design, weights = nonnegative_reconstruction(representation_design, values)
                residual = np.rint(values - design @ weights).astype(np.int32)
                residual = residual.reshape(n_samples, n_time, len(group)).transpose(0, 2, 1)
                reconstructed_bits = representation_bits + model_bits + temporal_bits_raw(residual)

                target_bits += direct_bits
                shared_target_bits += reconstructed_bits
                gain_bits += direct_bits - reconstructed_bits

            curve_rows.append({
                "group": group,
                "tau": int(tau),
                "gain_bits": float(gain_bits),
                "target_bits": float(target_bits),
                "shared_target_bits": float(shared_target_bits),
            })

        curve = pd.DataFrame(curve_rows)
        gain0 = float(curve.loc[curve["tau"] == 0, "gain_bits"].iloc[0])
        best_index = curve["gain_bits"].idxmax()
        best_lag = int(curve.loc[best_index, "tau"])
        gain_star = float(curve.loc[best_index, "gain_bits"])
        gain_min = float(curve["gain_bits"].min())
        curve["alignment_cost_bits"] = gain_star - curve["gain_bits"]
        curve["loss_from_zero_bits"] = gain0 - curve["gain_bits"]
        nonzero = curve[curve["tau"] != 0]
        # A nominal correction can consume the entire available lag range.
        # Such a fold still has a valid zero-lag diagnostic, but there is no
        # nonzero cost over which to average. Its empty-range cost is zero.
        mean_cost = 0.0 if nonzero.empty else float(nonzero["alignment_cost_bits"].mean())
        maximum_cost = 0.0 if nonzero.empty else float(nonzero["alignment_cost_bits"].max())

        group_rows.append({
            "group": group,
            "gain_at_zero_bits": gain0,
            "maximum_gain_bits": gain_star,
            "minimum_gain_bits": gain_min,
            "gain_range_bits": gain_star - gain_min,
            "cost_at_zero_bits": gain_star - gain0,
            "mean_alignment_cost_bits": mean_cost,
            "maximum_alignment_cost_bits": maximum_cost,
            "best_lag": best_lag,
            "zero_is_best": best_lag == 0,
        })
        curve["gain_at_zero_bits"] = gain0
        curve["maximum_gain_bits"] = gain_star
        curve["best_lag"] = best_lag
        all_curves.append(curve)

    group_summary = pd.DataFrame(group_rows)
    lag_curves = pd.concat(all_curves, ignore_index=True)
    selected = group_summary.loc[group_summary["mean_alignment_cost_bits"].idxmax()]
    result = {
        "score_bits": float(selected["mean_alignment_cost_bits"]),
        "group": selected["group"],
        "best_lag": int(selected["best_lag"]),
        "zero_is_best": bool(selected["zero_is_best"]),
        "gain_at_zero_bits": float(selected["gain_at_zero_bits"]),
        "maximum_gain_bits": float(selected["maximum_gain_bits"]),
        "lags": lags,
    }
    if return_details:
        return result, {
            "group_summary": group_summary,
            "lag_curves": lag_curves,
            "quantized": Q,
        }
    return result


def groups_for_metric(config, n_channels):
    all_channels = set(range(n_channels))
    groups = {}
    for group_id, channels in config.sensor_groups_indexed().items():
        channels = set(channels)
        if channels and channels != all_channels:
            groups[group_id] = channels
    return groups


def compute_smdl_for_dataset(
    config,
    fold: int,
    split: str,
    lags: list[int],
    max_samples: int | None = None,
    seed: int = 42,
    with_labels: bool = False,
    **metric_kwargs,
):
    train = build_dataset(config, fold, "train", lags=lags, normalize=False)
    if split == "train":
        ds = train
    elif split == "test":
        ds = build_dataset(config, fold, "test", lags=lags, normalize=False)
    else:
        test = build_dataset(config, fold, "test", lags=lags, normalize=False)
        indices = np.sort(np.concatenate([train._indices, test._indices]))
        if len(indices) != len(np.unique(indices)):
            raise ValueError("train and test indices overlap")
        return _compute_smdl_arrays(
            config,
            np.asarray(train._X[indices], dtype=np.float32),
            train._all_labels[indices] if with_labels else None,
            lags,
            max_samples,
            seed,
            fixed_window_size=train.window_size,
            circular=False,
            **metric_kwargs,
        )

    x, y = raw_windows_numpy(ds)
    return _compute_smdl_arrays(
        config,
        x,
        y if with_labels else None,
        lags,
        max_samples,
        seed,
        fixed_window_size=ds.window_size,
        circular=False,
        **metric_kwargs,
    )


def _compute_smdl_arrays(
    config,
    x,
    y,
    lags,
    max_samples,
    seed,
    group_names=None,
    **metric_kwargs,
):
    if max_samples is not None and len(x) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(x), size=max_samples, replace=False))
        x = x[indices]
        if y is not None:
            y = y[indices]

    groups = groups_for_metric(config, x.shape[1])
    if group_names is not None:
        group_ids = [config.group_index(name) for name in group_names]
        groups = {group_id: groups[group_id] for group_id in group_ids}

    result, details = mdl_alignment_loss(
        x,
        y,
        lags=lags,
        groups=groups,
        return_details=True,
        **metric_kwargs,
    )
    group_name_by_repr = {
        repr(tuple(channels)): name for name, channels in config.sensor_groups.items()
    }
    group_summary = details["group_summary"].copy()
    lag_curves = details["lag_curves"].copy()
    for frame in (group_summary, lag_curves):
        frame["group_name"] = frame["group"].map(
            lambda group: group_name_by_repr.get(repr(tuple(group)), repr(group))
        )

    return result, group_summary, lag_curves

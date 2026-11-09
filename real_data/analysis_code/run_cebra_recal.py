#!/usr/bin/env python3
"""
Standalone CEBRA 3D embedding for NC40017 session pickles (default: 250703).

Bins good-unit spikes into 1 s firing rates, fits CEBRA, optionally masks by
derived angular velocity and removes outliers, then saves .npy + SVG/HTML plots.

Example
-------
    python run_cebra_recal.py
    python run_cebra_recal.py --use-velocity-mask true --remove-outliers true
    # Default plots (one fit): time, landmark_frame_angle, sound_frame_angle
    python run_cebra_recal.py --color-plots time,landmark_frame_angle,sound_frame_angle

Paths and toggles live in CONFIG below and can be overridden via argparse.

Environment note
----------------
Unit files are pandas DataFrame pickles. An environment needs both a pandas
that can read them (often >= 2.0) and cebra. If read_pickle fails, this script
also accepts plain ``pickle`` payloads that are ``dict`` of column -> array.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re

import numpy as np
from packaging.version import Version
from scipy import stats
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3d projection

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "pandas is required. Install a pandas version that can unpickle the "
        "session DataFrames (pandas >= 2.0 recommended)."
    ) from exc

try:
    import cebra
except ImportError as exc:
    raise SystemExit(
        "cebra is required. Activate an environment that has both a compatible "
        "pandas and cebra (see README / plan notes about the env split)."
    ) from exc

try:
    import plotly.graph_objs as go
except ImportError as exc:
    raise SystemExit("plotly is required for interactive HTML plots.") from exc


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
_DEFAULT_DATA_ROOT = (
    "/Users/devenshidfar/Desktop/PhD/Skylar_data/NC40017_data_share"
)

CONFIG = {
    "data_root": _DEFAULT_DATA_ROOT,
    "session": "250703",
    "bin_size": 1.0,
    "output_dimension": 3,
    "temperature": 1.5,
    "max_iterations": 1000,
    "batch_size": 512,
    "use_velocity_mask": False,
    "vel_threshold": 5.0,  # deg/s on |Δ degree_lab| / bin_size
    "include_pre_session": False,
    "remove_outliers": False,
    "outlier_params": {
        "pct_distance": 1,
        "pct_neighbors": 65,
        "pct_dist": 70,
    },
    "mark_outliers": False,
    # One CEBRA fit; one SVG/HTML pair per entry. Built-ins:
    #   time, landmark_frame_angle (degree_landmark % 360),
    #   sound_frame_angle (degree_sound % 360). Also accepts raw pkl columns.
    "color_plots": ["time", "landmark_frame_angle", "sound_frame_angle"],
    "save_dir": None,  # default: <repo>/results/<session>/cebra_3d
    "animal_prefix": "NC40017",
    "min_bins": 201,
}

BEHAVIOR_COLUMNS = ("degree_lab", "degree_sound", "degree_landmark")
REQUIRED_UNIT_COLUMNS = ("elapsed_sec", "degree_lab", "degree_sound", "degree_landmark")


# ---------------------------------------------------------------------------
# SVG helpers (Affinity-safe)
# ---------------------------------------------------------------------------

def patch_svg_for_affinity_designer(svg_path):
    """Fix matplotlib SVG text positioning for Affinity Designer."""
    if Version(matplotlib.__version__) >= Version("3.10.0"):
        return

    with open(svg_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _fix_use_tag(match):
        tag = match.group(0)
        x_match = re.search(r'\bx="([^"]*)"', tag)
        y_match = re.search(r'\by="([^"]*)"', tag)
        if not x_match and not y_match:
            return tag

        x = x_match.group(1) if x_match else "0"
        y = y_match.group(1) if y_match else "0"
        tag = re.sub(r'\s*x="[^"]*"', "", tag)
        tag = re.sub(r'\s*y="[^"]*"', "", tag)

        translate = f'transform="translate({x} {y})"'
        if 'transform="' in tag:
            tag = re.sub(
                r'transform="([^"]*)"',
                lambda m: f'transform="translate({x} {y}) {m.group(1)}"',
                tag,
            )
        elif tag.endswith("/>"):
            tag = tag[:-2] + f" {translate}/>"
        else:
            tag = tag.replace(">", f" {translate}>", 1)
        return tag

    patched = re.sub(r"<use\b[^>]*/>", _fix_use_tag, content)
    if patched != content:
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(patched)


def save_svg(save_path, fig=None, **kwargs):
    """Save a figure as SVG with Affinity Designer-compatible text positioning."""
    if fig is None:
        fig = plt.gcf()
    kwargs.setdefault("format", "svg")
    fig.savefig(save_path, **kwargs)
    patch_svg_for_affinity_designer(save_path)


# ---------------------------------------------------------------------------
# Interpolation / outlier helpers
# ---------------------------------------------------------------------------

def linear_interpolate_nans_1d(array_1d):
    """Replace NaNs in a 1D array by linear interpolation along the index."""
    array_1d = np.asarray(array_1d, dtype=float).copy()
    x = np.arange(len(array_1d))
    good = ~np.isnan(array_1d)
    if not np.any(good):
        return array_1d
    array_1d[~good] = np.interp(x[~good], x[good], array_1d[good])
    return array_1d


def linear_interpolate_nans_2d(array_2d):
    """Replace NaNs with linearly interpolated values (per column)."""
    array_2d = np.asarray(array_2d, dtype=float).copy()
    x = np.arange(len(array_2d))
    for dim in range(array_2d.shape[1]):
        col = array_2d[:, dim]
        good = ~np.isnan(col)
        if not np.any(good):
            continue
        array_2d[~good, dim] = np.interp(x[~good], x[good], col[good])
    return array_2d


def expand_embeddings_to_full_trial(
    embeddings_high_vel, n_bins, high_vel_mask, interp=True
):
    """Place high-velocity embeddings into a full-length array; optionally interp."""
    full_embeddings = np.full(
        (n_bins, embeddings_high_vel.shape[1]), np.nan, dtype=float
    )
    full_embeddings[high_vel_mask, :] = embeddings_high_vel
    if interp:
        return linear_interpolate_nans_2d(full_embeddings)
    return full_embeddings


def nt_TDA_mask(data, pct_distance=1, pct_neighbors=50, pct_dist=80, verbose=False):
    """
    Detect outliers with the TDA-inspired method from run_manifold_H.py.
    Returns a boolean mask: True = inlier, False = outlier.
    Builds a full N x N distance matrix.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    print(
        f"[WARNING] nt_TDA_mask allocates a full {n} x {n} distance matrix "
        f"(~{n * n * 8 / 1e9:.2f} GB float64)."
    )
    if n >= 5000:
        print(
            f"[WARNING] N={n} is large; outlier detection may be slow or memory-heavy."
        )

    if verbose:
        print(f"[DEBUG] Running nt_TDA_mask on data of shape {data.shape}")
        print(
            f"[DEBUG] Using params: pct_distance={pct_distance}, "
            f"pct_neighbors={pct_neighbors}, pct_dist={pct_dist}"
        )

    distances = distance_matrix(data, data)
    np.fill_diagonal(distances, 10)

    neighborhood_radius = np.percentile(distances, pct_distance, axis=0)
    neighbor_counts = np.sum(distances <= neighborhood_radius[:, None], axis=1)
    threshold_neighbors = np.percentile(neighbor_counts, pct_neighbors)
    outlier_indices_1 = np.where(neighbor_counts < threshold_neighbors)[0]

    neighbgraph = NearestNeighbors(n_neighbors=5).fit(distances)
    dists, _ = neighbgraph.kneighbors(distances)
    min_distance_to_any_point = np.mean(dists, axis=1)
    distance_threshold = np.percentile(min_distance_to_any_point, pct_dist)
    outlier_indices_2 = np.where(min_distance_to_any_point > distance_threshold)[0]

    all_outliers = np.unique(np.concatenate([outlier_indices_1, outlier_indices_2]))
    mask = np.ones(n, dtype=bool)
    mask[all_outliers] = False

    if verbose:
        print(
            f"[DEBUG] inliers={int(mask.sum())}/{n}, "
            f"outlier ratio={1.0 - mask.sum() / n:.3f}"
        )
    return mask


# ---------------------------------------------------------------------------
# Data loading / binning
# ---------------------------------------------------------------------------

def _unit_pkl_path(session_dir, animal_prefix, session, unit_idx):
    return os.path.join(
        session_dir, f"{animal_prefix}_{session}_unit_{int(unit_idx)}.pkl"
    )


def _load_unit_frame(path):
    """
    Load a unit pickle as a DataFrame.

    Prefer pandas.read_pickle. If that fails (common when an older pandas
    cannot decode a newer BlockManager pickle), fall back to pickle.load for
    plain dict-of-arrays / dict-of-lists payloads.
    """
    try:
        df = pd.read_pickle(path)
        if isinstance(df, pd.DataFrame):
            return df
        raise TypeError(f"Expected DataFrame from read_pickle, got {type(df)}")
    except Exception as pandas_exc:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            raise RuntimeError(
                f"Failed to unpickle {path}. This usually means the active "
                f"pandas ({pd.__version__}) cannot read a newer DataFrame "
                f"pickle. Use pandas >= 2.0 in an environment that also has "
                f"cebra, or re-save units as a plain dict of arrays."
            ) from pandas_exc

        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
        raise RuntimeError(
            f"Failed to unpickle {path}: unsupported payload type {type(obj)}. "
            f"pandas error was: {pandas_exc}"
        ) from pandas_exc


def load_unit_dataframes(cfg):
    """Load units_indices.npy and the matching unit pickles in index order."""
    session_dir = os.path.join(cfg["data_root"], cfg["session"])
    indices_path = os.path.join(session_dir, "units_indices.npy")
    if not os.path.isfile(indices_path):
        raise FileNotFoundError(f"Missing units_indices.npy: {indices_path}")

    unit_indices = np.load(indices_path)
    print(f"[INFO] units_indices shape={unit_indices.shape}, values={unit_indices}")

    frames = []
    for unit_idx in unit_indices:
        path = _unit_pkl_path(
            session_dir, cfg["animal_prefix"], cfg["session"], unit_idx
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing unit pickle for index {unit_idx}: {path}")
        df = _load_unit_frame(path)

        if len(df) == 0:
            raise ValueError(f"Unit {unit_idx} has no spikes: {path}")
        for col in REQUIRED_UNIT_COLUMNS:
            if col not in df.columns:
                raise KeyError(f"Unit {unit_idx} missing required column '{col}'")
        for color_key in cfg["color_plots"]:
            if color_key in (
                "time",
                "landmark_frame_angle",
                "sound_frame_angle",
            ):
                continue
            if color_key not in df.columns:
                raise KeyError(
                    f"color_plots entry '{color_key}' not found in unit "
                    f"{unit_idx} columns {list(df.columns)}"
                )
        print(
            f"[INFO] loaded unit {unit_idx}: nrows={len(df)}, "
            f"elapsed=({df['elapsed_sec'].min():.4f}, {df['elapsed_sec'].max():.4f})"
        )
        frames.append(df)

    return unit_indices, frames


def resolve_time_window(frames, cfg):
    """Return t_start, t_end from spike elapsed_sec across loaded units."""
    elapsed_min = min(float(df["elapsed_sec"].min()) for df in frames)
    elapsed_max = max(float(df["elapsed_sec"].max()) for df in frames)
    t_end = elapsed_max
    if cfg["include_pre_session"]:
        t_start = elapsed_min
    else:
        t_start = 0.0
    print(
        f"[INFO] elapsed_sec global min={elapsed_min:.4f}, max={elapsed_max:.4f}; "
        f"using t_start={t_start:.4f}, t_end={t_end:.4f} "
        f"(include_pre_session={cfg['include_pre_session']})"
    )
    return t_start, t_end, elapsed_min, elapsed_max


def build_bins(t_start, t_end, bin_size, min_bins):
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    n_bins = len(bins) - 1
    if n_bins < min_bins:
        raise ValueError(
            f"Only {n_bins} bins of size {bin_size}s "
            f"(need >= {min_bins}). t_start={t_start}, t_end={t_end}."
        )
    print(f"[INFO] bins: n_edges={len(bins)}, n_bins={n_bins}, bin_size={bin_size}")
    return bins


def bin_firing_rates(frames, bins, bin_size):
    """Count spikes per bin per unit; return Hz array (n_bins, n_units)."""
    rates = []
    for i, df in enumerate(frames):
        times = df["elapsed_sec"].to_numpy(dtype=float)
        counts, _, _ = stats.binned_statistic(
            times, np.ones_like(times), statistic="sum", bins=bins
        )
        counts = np.nan_to_num(counts, nan=0.0)
        rates.append(counts / bin_size)
        print(
            f"[INFO] unit col {i}: spikes={len(times)}, "
            f"rate mean={rates[-1].mean():.3f} Hz, "
            f"empty_bins={int(np.sum(counts == 0))}"
        )
    if not rates:
        neural_data = np.empty((len(bins) - 1, 0))
    else:
        neural_data = np.stack(rates, axis=1)
    print(f"[INFO] neural_data shape={neural_data.shape} (n_bins, n_units)")
    return neural_data


def _unwrap_degrees(values_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(values_deg, dtype=float))))


def _wrap_deg_360(values_deg):
    """Wrap degrees into [0, 360)."""
    return np.mod(np.asarray(values_deg, dtype=float), 360.0)


def bin_behavior(frames, bins, color_plots):
    """
    Pool spike rows across units and mean-bin behavior columns.

    Unwrap degree_landmark along sorted spike time before binning (wrapped
    circular mean is unsafe). Frame-angle color channels are then wrapped
    back to [0, 360).
    """
    elapsed_list = []
    lab_list = []
    sound_list = []
    landmark_list = []
    extra_cols = [
        c
        for c in color_plots
        if c
        not in (
            "time",
            "landmark_frame_angle",
            "sound_frame_angle",
            "degree_lab",
            "degree_sound",
            "degree_landmark",
        )
    ]
    extra_lists = {c: [] for c in extra_cols}

    for df in frames:
        elapsed_list.append(df["elapsed_sec"].to_numpy(dtype=float))
        lab_list.append(df["degree_lab"].to_numpy(dtype=float))
        sound_list.append(df["degree_sound"].to_numpy(dtype=float))
        landmark_list.append(df["degree_landmark"].to_numpy(dtype=float))
        for c in extra_cols:
            extra_lists[c].append(df[c].to_numpy(dtype=float))

    elapsed = np.concatenate(elapsed_list)
    lab = np.concatenate(lab_list)
    sound = np.concatenate(sound_list)
    landmark = np.concatenate(landmark_list)

    order = np.argsort(elapsed)
    elapsed_s = elapsed[order]
    lab_s = lab[order]
    sound_s = sound[order]
    landmark_unwrapped = _unwrap_degrees(landmark[order])

    binned = {}
    for name, values in (
        ("degree_lab", lab_s),
        ("degree_sound", sound_s),
        ("degree_landmark", landmark_unwrapped),
    ):
        mean_vals, _, _ = stats.binned_statistic(
            elapsed_s, values, statistic="mean", bins=bins
        )
        n_nan = int(np.isnan(mean_vals).sum())
        print(f"[INFO] binned {name}: shape={mean_vals.shape}, nan_bins={n_nan}")
        binned[name] = linear_interpolate_nans_1d(mean_vals)

    for c in extra_cols:
        raw = np.concatenate(extra_lists[c])[order]
        mean_vals, _, _ = stats.binned_statistic(
            elapsed_s, raw, statistic="mean", bins=bins
        )
        n_nan = int(np.isnan(mean_vals).sum())
        print(f"[INFO] binned extra color '{c}': nan_bins={n_nan}")
        binned[c] = linear_interpolate_nans_1d(mean_vals)

    n_bins = len(bins) - 1
    color_channels = {}
    color_labels = {}
    for key in color_plots:
        if key == "time":
            color_channels[key] = np.arange(n_bins, dtype=float)
            color_labels[key] = "time (bin index)"
        elif key == "landmark_frame_angle":
            color_channels[key] = _wrap_deg_360(binned["degree_landmark"])
            color_labels[key] = "landmark frame angle (deg)"
        elif key == "sound_frame_angle":
            color_channels[key] = _wrap_deg_360(binned["degree_sound"])
            color_labels[key] = "sound / audio frame angle (deg)"
        elif key in binned:
            color_channels[key] = binned[key].copy()
            color_labels[key] = key
        else:
            raise KeyError(f"Unknown color_plots entry: {key!r}")
        print(
            f"[INFO] color channel '{key}': "
            f"min={np.nanmin(color_channels[key]):.3f}, "
            f"max={np.nanmax(color_channels[key]):.3f}"
        )

    return binned, color_channels, color_labels


def derive_angular_velocity(binned_degree_lab, bin_size, vel_threshold):
    """binned_vel[0]=0; later bins = |Δ degree_lab| / bin_size (deg/s)."""
    binned_vel = np.zeros_like(binned_degree_lab, dtype=float)
    binned_vel[1:] = np.abs(np.diff(binned_degree_lab)) / bin_size
    high_vel_mask = binned_vel > vel_threshold
    finite = binned_vel[np.isfinite(binned_vel)]
    print(
        f"[INFO] binned_vel (deg/s): min={finite.min():.3f}, "
        f"median={np.median(finite):.3f}, max={finite.max():.3f}, "
        f"n_high={int(high_vel_mask.sum())}/{len(high_vel_mask)} "
        f"(threshold={vel_threshold})"
    )
    return binned_vel, high_vel_mask


# ---------------------------------------------------------------------------
# CEBRA
# ---------------------------------------------------------------------------

def fit_cebra(
    neural_data,
    high_vel_mask,
    use_velocity_mask,
    output_dimension,
    temperature,
    max_iterations,
    batch_size,
):
    """Fit CEBRA; optionally only on high-velocity bins and expand + interp."""
    model = cebra.CEBRA(
        output_dimension=output_dimension,
        max_iterations=max_iterations,
        batch_size=batch_size,
        temperature=temperature,
    )

    embeddings_low_vel = None

    if use_velocity_mask:
        neural_fit = neural_data[high_vel_mask]
        neural_low = neural_data[~high_vel_mask]
        print(
            f"[INFO] velocity mask ON: fit on {neural_fit.shape[0]} high-vel bins; "
            f"{neural_low.shape[0]} low-vel bins"
        )
        if neural_fit.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 high-velocity bins to fit CEBRA; got {neural_fit.shape[0]}"
            )
        model.fit(neural_fit)
        embeddings_high = model.transform(neural_fit)
        if neural_low.shape[0] >= 1:
            embeddings_low_vel = model.transform(neural_low)
        embeddings = expand_embeddings_to_full_trial(
            embeddings_high, neural_data.shape[0], high_vel_mask, interp=True
        )
    else:
        print(f"[INFO] velocity mask OFF: fit/transform on all {neural_data.shape[0]} bins")
        model.fit(neural_data)
        embeddings = model.transform(neural_data)
        high_vel_mask = np.ones(neural_data.shape[0], dtype=bool)

    print(f"[INFO] embeddings shape={embeddings.shape}")
    if embeddings_low_vel is not None:
        print(f"[INFO] embeddings_low_vel shape={embeddings_low_vel.shape}")
    return embeddings, embeddings_low_vel, high_vel_mask


def apply_outlier_removal(embeddings, binned_vel, color_channels, outlier_params):
    """nt_TDA_mask → NaN outliers → linear interp on embeddings and behavior."""
    inlier_mask = nt_TDA_mask(
        embeddings,
        pct_distance=outlier_params.get("pct_distance", 1),
        pct_neighbors=outlier_params.get("pct_neighbors", 65),
        pct_dist=outlier_params.get("pct_dist", 70),
        verbose=True,
    )
    outlier_mask = ~inlier_mask
    n_out = int(outlier_mask.sum())
    print(f"[INFO] outliers={n_out}/{len(inlier_mask)}")

    embeddings_pre = embeddings.copy()
    embeddings = embeddings.copy()
    embeddings[outlier_mask, :] = np.nan
    embeddings = linear_interpolate_nans_2d(embeddings)

    binned_vel = binned_vel.copy()
    binned_vel[outlier_mask] = np.nan
    binned_vel = linear_interpolate_nans_1d(binned_vel)

    patched_colors = {}
    for key, vals in color_channels.items():
        arr = np.asarray(vals, dtype=float).copy()
        arr[outlier_mask] = np.nan
        patched_colors[key] = linear_interpolate_nans_1d(arr)

    outlier_coords = embeddings_pre[outlier_mask] if n_out else None
    return embeddings, binned_vel, patched_colors, inlier_mask, outlier_coords


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_embeddings_static(
    embeddings,
    color_vals,
    color_label,
    save_path,
    embeddings_low_vel=None,
    outlier_coords=None,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        embeddings[:, 2],
        c=color_vals,
        cmap="jet",
        s=5,
        alpha=0.7,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.7, label=color_label)

    if embeddings_low_vel is not None and len(embeddings_low_vel) > 0:
        ax.scatter(
            embeddings_low_vel[:, 0],
            embeddings_low_vel[:, 1],
            embeddings_low_vel[:, 2],
            color="magenta",
            s=10,
            label="low velocity",
            alpha=0.8,
        )
    if outlier_coords is not None and len(outlier_coords) > 0:
        ax.scatter(
            outlier_coords[:, 0],
            outlier_coords[:, 1],
            outlier_coords[:, 2],
            color="black",
            s=12,
            label="outliers",
            alpha=0.9,
        )
    if embeddings_low_vel is not None or outlier_coords is not None:
        ax.legend()

    ax.set_title(f"CEBRA 3D embeddings ({color_label})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    save_svg(save_path, fig=fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved static SVG: {save_path}")


def plot_embeddings_interactive(
    embeddings,
    color_vals,
    color_label,
    save_path,
    embeddings_low_vel=None,
    outlier_coords=None,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=color_vals,
                colorscale="Jet",
                opacity=0.7,
                colorbar=dict(title=color_label),
            ),
            name="embeddings",
        )
    )
    if embeddings_low_vel is not None and len(embeddings_low_vel) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=embeddings_low_vel[:, 0],
                y=embeddings_low_vel[:, 1],
                z=embeddings_low_vel[:, 2],
                mode="markers",
                marker=dict(size=4, color="magenta", opacity=0.8),
                name="low velocity",
            )
        )
    if outlier_coords is not None and len(outlier_coords) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=outlier_coords[:, 0],
                y=outlier_coords[:, 1],
                z=outlier_coords[:, 2],
                mode="markers",
                marker=dict(size=4, color="black", opacity=0.9),
                name="outliers",
            )
        )
    fig.update_layout(
        title=f"CEBRA 3D embeddings ({color_label})",
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
    )
    fig.write_html(save_path)
    print(f"[INFO] saved interactive HTML: {save_path}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def parse_args(defaults):
    parser = argparse.ArgumentParser(
        description="Run standalone CEBRA on NC40017 session unit pickles."
    )
    parser.add_argument("--data-root", default=defaults["data_root"])
    parser.add_argument("--session", default=defaults["session"])
    parser.add_argument("--save-dir", default=defaults["save_dir"])
    parser.add_argument("--bin-size", type=float, default=defaults["bin_size"])
    parser.add_argument(
        "--output-dimension", type=int, default=defaults["output_dimension"]
    )
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument(
        "--max-iterations", type=int, default=defaults["max_iterations"]
    )
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument(
        "--use-velocity-mask",
        type=_bool_arg,
        default=defaults["use_velocity_mask"],
    )
    parser.add_argument(
        "--vel-threshold", type=float, default=defaults["vel_threshold"]
    )
    parser.add_argument(
        "--include-pre-session",
        type=_bool_arg,
        default=defaults["include_pre_session"],
    )
    parser.add_argument(
        "--remove-outliers", type=_bool_arg, default=defaults["remove_outliers"]
    )
    parser.add_argument(
        "--mark-outliers", type=_bool_arg, default=defaults["mark_outliers"]
    )
    parser.add_argument(
        "--color-plots",
        default=",".join(defaults["color_plots"]),
        help=(
            "Comma-separated color channels for plots after one CEBRA fit. "
            "Built-ins: time, landmark_frame_angle, sound_frame_angle. "
            "Also accepts pkl columns (e.g. degree_lab)."
        ),
    )
    return parser.parse_args()


def _parse_color_plots(value):
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = [part.strip() for part in str(value).split(",")]
    items = [x for x in items if x]
    if not items:
        raise ValueError("color_plots must contain at least one entry")
    return items


def config_from_args(args):
    cfg = dict(CONFIG)
    cfg["data_root"] = args.data_root
    cfg["session"] = str(args.session)
    cfg["bin_size"] = float(args.bin_size)
    cfg["output_dimension"] = int(args.output_dimension)
    cfg["temperature"] = float(args.temperature)
    cfg["max_iterations"] = int(args.max_iterations)
    cfg["batch_size"] = int(args.batch_size)
    cfg["use_velocity_mask"] = bool(args.use_velocity_mask)
    cfg["vel_threshold"] = float(args.vel_threshold)
    cfg["include_pre_session"] = bool(args.include_pre_session)
    cfg["remove_outliers"] = bool(args.remove_outliers)
    cfg["mark_outliers"] = bool(args.mark_outliers)
    cfg["color_plots"] = _parse_color_plots(args.color_plots)
    if args.save_dir:
        cfg["save_dir"] = args.save_dir
    else:
        cfg["save_dir"] = os.path.join(
            _REPO_ROOT, "results", cfg["session"], "cebra_3d"
        )
    return cfg


def run_tag(cfg):
    """Tag shared by arrays from one fit (color is only in plot filenames)."""
    return (
        f"temp{cfg['temperature']:g}"
        f"_vel{int(cfg['use_velocity_mask'])}"
        f"_out{int(cfg['remove_outliers'])}"
    )


def color_slug(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(name))


def main():
    args = parse_args(CONFIG)
    cfg = config_from_args(args)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    tag = run_tag(cfg)
    print(f"[INFO] run tag={tag}")
    print(f"[INFO] save_dir={cfg['save_dir']}")
    print(f"[INFO] pandas={pd.__version__}, cebra={getattr(cebra, '__version__', '?')}")

    unit_indices, frames = load_unit_dataframes(cfg)
    t_start, t_end, elapsed_min, elapsed_max = resolve_time_window(frames, cfg)
    bins = build_bins(t_start, t_end, cfg["bin_size"], cfg["min_bins"])

    neural_data = bin_firing_rates(frames, bins, cfg["bin_size"])
    binned_beh, color_channels, color_labels = bin_behavior(
        frames, bins, cfg["color_plots"]
    )
    binned_vel, high_vel_mask = derive_angular_velocity(
        binned_beh["degree_lab"], cfg["bin_size"], cfg["vel_threshold"]
    )

    if neural_data.shape[0] != len(binned_vel):
        raise RuntimeError(
            f"Bin count mismatch: neural {neural_data.shape[0]} vs vel {len(binned_vel)}"
        )

    embeddings, embeddings_low_vel, high_vel_mask = fit_cebra(
        neural_data=neural_data,
        high_vel_mask=high_vel_mask,
        use_velocity_mask=cfg["use_velocity_mask"],
        output_dimension=cfg["output_dimension"],
        temperature=cfg["temperature"],
        max_iterations=cfg["max_iterations"],
        batch_size=cfg["batch_size"],
    )

    inlier_mask = np.ones(embeddings.shape[0], dtype=bool)
    outlier_coords = None
    if cfg["remove_outliers"]:
        (
            embeddings,
            binned_vel,
            color_channels,
            inlier_mask,
            outlier_coords,
        ) = apply_outlier_removal(
            embeddings, binned_vel, color_channels, cfg["outlier_params"]
        )
        if not cfg["mark_outliers"]:
            outlier_coords = None
    else:
        print("[INFO] remove_outliers=False; skipping nt_TDA_mask")

    # Save arrays (shared across color plots)
    emb_path = os.path.join(cfg["save_dir"], f"embedding_{tag}.npy")
    neural_path = os.path.join(cfg["save_dir"], f"neural_data_{tag}.npy")
    vel_path = os.path.join(cfg["save_dir"], f"binned_vel_{tag}.npy")
    hv_path = os.path.join(cfg["save_dir"], f"high_vel_mask_{tag}.npy")
    in_path = os.path.join(cfg["save_dir"], f"inlier_mask_{tag}.npy")
    np.save(emb_path, embeddings)
    np.save(neural_path, neural_data)
    np.save(vel_path, binned_vel)
    np.save(hv_path, high_vel_mask)
    np.save(in_path, inlier_mask)
    print(f"[INFO] saved {emb_path}")
    print(f"[INFO] saved {neural_path}")
    print(f"[INFO] saved {vel_path}")
    print(f"[INFO] saved {hv_path}")
    print(f"[INFO] saved {in_path}")

    if embeddings_low_vel is not None:
        low_path = os.path.join(cfg["save_dir"], f"embeddings_low_vel_{tag}.npy")
        np.save(low_path, embeddings_low_vel)
        print(f"[INFO] saved {low_path}")

    plot_files = {}
    for key in cfg["color_plots"]:
        slug = color_slug(key)
        svg_path = os.path.join(
            cfg["save_dir"], f"embedding_3d_{tag}_color-{slug}.svg"
        )
        html_path = os.path.join(
            cfg["save_dir"], f"embedding_3d_{tag}_color-{slug}.html"
        )
        color_npy = os.path.join(
            cfg["save_dir"], f"color_{slug}_{tag}.npy"
        )
        np.save(color_npy, color_channels[key])
        plot_embeddings_static(
            embeddings,
            color_channels[key],
            color_labels[key],
            svg_path,
            embeddings_low_vel=embeddings_low_vel,
            outlier_coords=outlier_coords,
        )
        plot_embeddings_interactive(
            embeddings,
            color_channels[key],
            color_labels[key],
            html_path,
            embeddings_low_vel=embeddings_low_vel,
            outlier_coords=outlier_coords,
        )
        plot_files[key] = {
            "svg": os.path.basename(svg_path),
            "html": os.path.basename(html_path),
            "color_npy": os.path.basename(color_npy),
            "label": color_labels[key],
        }

    meta = {
        "config": {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in cfg.items()
        },
        "unit_indices": unit_indices.tolist(),
        "t_start": float(t_start),
        "t_end": float(t_end),
        "elapsed_sec_min": float(elapsed_min),
        "elapsed_sec_max": float(elapsed_max),
        "n_bins": int(neural_data.shape[0]),
        "n_units": int(neural_data.shape[1]),
        "neural_data_shape": list(neural_data.shape),
        "embeddings_shape": list(embeddings.shape),
        "n_high_vel": int(np.sum(high_vel_mask)),
        "n_low_vel": int(np.sum(~high_vel_mask)),
        "n_inliers": int(np.sum(inlier_mask)),
        "n_outliers": int(np.sum(~inlier_mask)),
        "color_plots": cfg["color_plots"],
        "color_labels": color_labels,
        "run_tag": tag,
        "velocity_note": (
            "No velocity column in pickles; binned_vel is "
            "|Δ mean(degree_lab)| / bin_size in deg/s."
        ),
        "frame_angle_note": (
            "landmark_frame_angle = degree_landmark mean-binned after unwrap, "
            "then wrapped to [0, 360). sound_frame_angle = degree_sound "
            "mean-binned, then wrapped to [0, 360). Not audio frequency in Hz."
        ),
        "files": {
            "embedding": os.path.basename(emb_path),
            "neural_data": os.path.basename(neural_path),
            "binned_vel": os.path.basename(vel_path),
            "high_vel_mask": os.path.basename(hv_path),
            "inlier_mask": os.path.basename(in_path),
            "plots": plot_files,
        },
    }
    config_path = os.path.join(cfg["save_dir"], f"config_{tag}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] saved {config_path}")
    print("[INFO] done.")


if __name__ == "__main__":
    main()

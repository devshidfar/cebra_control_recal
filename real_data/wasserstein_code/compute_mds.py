import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
from scipy.io import loadmat
from packaging.version import Version

from sklearn.manifold import MDS

TARGET_RAT = 913  # set to None to include all rats
SI_SCORE_FIELD = "SI_score_hipp"  # or "SI_score_true"

file_path = '/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/results/all_rat_913/all_rat_913_all_sessions_data_backup.mat'
mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)

similarity_matrix = mat_data['pairwise_EMD_distance_matrix']
sessions = mat_data['sesions_for_EMD']
mean_H_error = mat_data['mean_H_error']
trial = mat_data['full_trial_1']
session_lookup = {session.session_idx: session for session in trial.sessions}

if 'rats_for_EMD' in mat_data:
    rat_ids = mat_data['rats_for_EMD']
else:
    rat_ids = np.array([session_lookup[session_id].rat for session_id in sessions])

si_scores = np.array([getattr(session_lookup[session_id], SI_SCORE_FIELD) for session_id in sessions])

save_dir = "wasserstein_plots"
if TARGET_RAT is not None:
    mask = rat_ids == TARGET_RAT
    if not np.any(mask):
        raise ValueError(f"No sessions found for TARGET_RAT={TARGET_RAT}")
    indices = np.where(mask)[0]
    similarity_matrix = similarity_matrix[np.ix_(indices, indices)]
    sessions = sessions[mask]
    mean_H_error = mean_H_error[mask]
    si_scores = si_scores[mask]
    rat_ids = rat_ids[mask]
    save_dir = f"wasserstein_plots/rat_{TARGET_RAT}"

os.makedirs(save_dir, exist_ok=True)

H_ERROR_THRESHOLD = 0.05
SI_SCORE_VMAX = 1.0
RAT_COLORS = {
    883: "tab:red",
    913: "tab:green",
    923: "tab:blue",
}


def patch_svg_for_affinity_designer(svg_path):
    """
    Fix matplotlib SVG text positioning for Affinity Designer.

    Affinity ignores x/y on <use> elements (matplotlib < 3.10); glyphs stack at
    the origin. Matplotlib 3.10+ emits transform=translate(...) instead.
    """
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


def _annotate_sessions(ax, embeddings, sessions, rat_ids, session_labels_only):
    for i in range(len(embeddings)):
        label = f"sess {sessions[i]}" if session_labels_only else f"rat {rat_ids[i]}, sess {sessions[i]}"
        ax.annotate(
            label,
            (embeddings[i, 0], embeddings[i, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
            color="black",
        )


def save_mds_2d_by_h_error(embeddings, sessions, rat_ids, mean_H_error, save_path, session_labels_only=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=mean_H_error,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=max(mean_H_error.max(), H_ERROR_THRESHOLD),
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean H error")
    _annotate_sessions(ax, embeddings, sessions, rat_ids, session_labels_only)

    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_title("2D MDS of Wasserstein distances (colored by H error)")
    save_svg(save_path, fig=fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_mds_2d_by_si(embeddings, sessions, rat_ids, si_scores, save_path, session_labels_only=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=si_scores,
        cmap="RdYlGn",
        vmin=0,
        vmax=SI_SCORE_VMAX,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("SI score")
    _annotate_sessions(ax, embeddings, sessions, rat_ids, session_labels_only)

    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_title("2D MDS of Wasserstein distances (colored by SI score)")
    save_svg(save_path, fig=fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_mds_2d_by_rat(embeddings, sessions, rat_ids, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for rat_id in sorted(np.unique(rat_ids)):
        mask = rat_ids == rat_id
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            color=RAT_COLORS.get(rat_id, "tab:gray"),
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
            label=f"Rat {rat_id}",
        )

    for i in range(len(embeddings)):
        ax.annotate(
            f"sess {sessions[i]}",
            (embeddings[i, 0], embeddings[i, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
            color="black",
        )

    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_title("2D MDS of Wasserstein distances (colored by rat)")
    ax.legend(title="Rat")
    save_svg(save_path, fig=fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


dimension = 2

if dimension == 2:
    n_components = 3
    mds = MDS(n_components=n_components, dissimilarity='precomputed')
    embeddings = mds.fit_transform(similarity_matrix)

    session_labels_only = TARGET_RAT is not None
    h_error_path = os.path.join(save_dir, "mds_2d_by_h_error.svg")
    si_path = os.path.join(save_dir, "mds_2d_by_si.svg")
    save_mds_2d_by_h_error(
        embeddings, sessions, rat_ids, mean_H_error, h_error_path,
        session_labels_only=session_labels_only,
    )
    save_mds_2d_by_si(
        embeddings, sessions, rat_ids, si_scores, si_path,
        session_labels_only=session_labels_only,
    )
    print(f"Saved H-error MDS plot to {h_error_path}")
    print(f"Saved SI-score MDS plot to {si_path}")

    if TARGET_RAT is None:
        rat_path = os.path.join(save_dir, "mds_2d_by_rat.svg")
        save_mds_2d_by_rat(embeddings, sessions, rat_ids, rat_path)
        print(f"Saved rat-colored MDS plot to {rat_path}")

elif dimension == 3:
    n_components = 3
    mds = MDS(n_components=n_components, dissimilarity='precomputed')
    embeddings = mds.fit_transform(similarity_matrix)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for rat_id in sorted(np.unique(rat_ids)):
        mask = rat_ids == rat_id
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            embeddings[mask, 2],
            color=RAT_COLORS.get(rat_id, "tab:gray"),
            alpha=0.6,
            label=f"Rat {rat_id}",
        )

    for i in range(len(embeddings[:, 0])):
        label = f"sess {sessions[i]}" if TARGET_RAT is not None else f"rat {rat_ids[i]}, sess {sessions[i]}"
        ax.text(
            embeddings[i, 0],
            embeddings[i, 1],
            embeddings[i, 2],
            label,
            color=RAT_COLORS.get(rat_ids[i], "black"),
            fontsize=8,
        )

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D MDS Scatter Plot')
    ax.legend(title="Rat")
    plt.show()


print(embeddings.shape)

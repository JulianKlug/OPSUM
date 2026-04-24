"""ROC AUC over the 72h admission timeline.

Computes per-timestep AUROC from raw test predictions, then generates a
two-panel figure: seaborn regplot (top) + positive event count (bottom).
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Shared computation
# ---------------------------------------------------------------------------

def compute_auroc_over_time(y_test_2d, y_prob_2d, n_bootstrap=1000, seed=42):
    """Compute per-timestep AUROC with bootstrap 95% CI.

    Args:
        y_test_2d: (n_subjects, n_timesteps) binary labels.
        y_prob_2d: (n_subjects, n_timesteps) predicted probabilities.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        timesteps: 1-d array of timestep indices (0-based).
        auroc_point: Point estimate of AUROC per timestep.
        auroc_lower: Lower bound of 95% CI.
        auroc_upper: Upper bound of 95% CI.
        n_positive: Number of positive cases per timestep.
    """
    rng = np.random.RandomState(seed)
    n_subjects, n_timesteps = y_test_2d.shape
    timesteps = np.arange(n_timesteps)

    auroc_point = np.full(n_timesteps, np.nan)
    auroc_lower = np.full(n_timesteps, np.nan)
    auroc_upper = np.full(n_timesteps, np.nan)
    n_positive = np.zeros(n_timesteps, dtype=int)

    for t in range(n_timesteps):
        y_t = y_test_2d[:, t]
        p_t = y_prob_2d[:, t]
        n_positive[t] = int(y_t.sum())

        # Need at least one positive and one negative to compute AUROC
        if n_positive[t] < 1 or n_positive[t] >= n_subjects:
            continue

        auroc_point[t] = roc_auc_score(y_t, p_t)

        # Bootstrap over subjects
        boot_aucs = np.full(n_bootstrap, np.nan)
        for b in range(n_bootstrap):
            idx = rng.randint(0, n_subjects, size=n_subjects)
            yb = y_t[idx]
            pb = p_t[idx]
            if yb.sum() < 1 or yb.sum() >= len(yb):
                continue
            boot_aucs[b] = roc_auc_score(yb, pb)

        valid = boot_aucs[~np.isnan(boot_aucs)]
        if len(valid) > 0:
            auroc_lower[t] = np.percentile(valid, 2.5)
            auroc_upper[t] = np.percentile(valid, 97.5)

    return timesteps, auroc_point, auroc_lower, auroc_upper, n_positive


# ---------------------------------------------------------------------------
# Two-panel regplot (AUROC + event count)
# ---------------------------------------------------------------------------

def plot_auroc_over_time(timesteps, auroc_point, n_positive):
    """Top: seaborn regplot of per-timestep AUROC; bottom: positive event count."""
    import seaborn as sns

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax_auc = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharex=ax_auc)

    # Top panel — regplot
    mask = ~np.isnan(auroc_point)
    t = timesteps[mask]
    auc = auroc_point[mask]

    sns.regplot(x=t, y=auc, ax=ax_auc, ci=95,
                scatter_kws={"s": 20, "alpha": 0.5, "color": "#7F8C8D"},
                line_kws={"color": "#2980B9", "linewidth": 1.5},
                color="#2980B9")
    ax_auc.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax_auc.set_ylabel("AUROC", fontsize=11)
    ax_auc.set_xlabel("")
    ax_auc.set_ylim(0, 1.05)
    ax_auc.spines["top"].set_visible(False)
    ax_auc.spines["right"].set_visible(False)
    plt.setp(ax_auc.get_xticklabels(), visible=False)

    # Bottom panel — positive count
    ax_bar.bar(timesteps, n_positive, width=1.0, color="#E74C3C", alpha=0.6)
    ax_bar.set_xlabel("Time after admission (hours)", fontsize=11)
    ax_bar.set_ylabel("# positive", fontsize=10)
    ax_bar.set_xlim(timesteps[0] - 0.5, timesteps[-1] + 0.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load predictions
    pred_path = (
        "/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/"
        "best_xgb_final_model/test_predictions.pkl"
    )
    with open(pred_path, "rb") as f:
        y_test, y_prob = pickle.load(f)

    # Reshape to (n_subjects, n_timesteps)
    n_subjects = 533
    n_timesteps = len(y_test) // n_subjects
    assert len(y_test) == n_subjects * n_timesteps, (
        f"Cannot reshape {len(y_test)} samples into ({n_subjects}, {n_timesteps})"
    )
    y_test_2d = y_test.reshape(n_subjects, n_timesteps)
    y_prob_2d = y_prob.reshape(n_subjects, n_timesteps)

    print(f"Data: {n_subjects} subjects x {n_timesteps} timesteps")
    print("Computing per-timestep AUROC with bootstrap CIs …")
    timesteps, auroc_pt, auroc_lo, auroc_hi, n_pos = compute_auroc_over_time(
        y_test_2d, y_prob_2d, n_bootstrap=1000,
    )
    print(f"AUROC range: {np.nanmin(auroc_pt):.3f} – {np.nanmax(auroc_pt):.3f}")

    # Output directory
    output_dir = (
        "/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/"
        "best_xgb_final_model/auroc_over_time_plots"
    )
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_auroc_over_time(timesteps, auroc_pt, n_pos)

    for fmt in ["png", "tiff"]:
        path = os.path.join(output_dir, f"auroc_over_time.{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

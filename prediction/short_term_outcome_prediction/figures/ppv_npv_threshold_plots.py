"""Generate PPV/NPV vs threshold plot with prediction distribution."""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def compute_ppv_npv(y_true, y_prob, thresholds):
    """Compute PPV and NPV at each threshold."""
    ppv = np.full_like(thresholds, np.nan)
    npv = np.full_like(thresholds, np.nan)
    sensitivity = np.full_like(thresholds, np.nan)
    specificity = np.full_like(thresholds, np.nan)

    for i, t in enumerate(thresholds):
        pred_pos = y_prob >= t
        pred_neg = ~pred_pos

        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        tn = np.sum(pred_neg & (y_true == 0))
        fn = np.sum(pred_neg & (y_true == 1))

        ppv[i] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv[i] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return ppv, npv, sensitivity, specificity


def plot_ppv_npv(y_true, y_prob, thresholds, ppv, npv, sensitivity, specificity,
                 smoothing_window=0):
    """Plot PPV/NPV with sensitivity/specificity and predicted probability histogram.

    Args:
        smoothing_window: If > 0, apply a Savitzky-Golay filter to smooth PPV and NPV curves.
            Recommended values: 51-101 for light smoothing, 151-201 for heavy smoothing.
            Must be odd. 0 = no smoothing (default).
    """
    from scipy.signal import savgol_filter

    if smoothing_window > 0:
        if smoothing_window % 2 == 0:
            smoothing_window += 1  # must be odd
        polyorder = min(3, smoothing_window - 1)
        ppv = savgol_filter(np.nan_to_num(ppv, nan=0.0), smoothing_window, polyorder)
        npv = savgol_filter(np.nan_to_num(npv, nan=1.0), smoothing_window, polyorder)
        ppv = np.clip(ppv, 0, 1)
        npv = np.clip(npv, 0, 1)

    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharex=ax_main)

    # Main plot: PPV and NPV
    ax_main.plot(thresholds, ppv, color="#E74C3C", linewidth=2.5, label="PPV")
    ax_main.plot(thresholds, npv, color="#2980B9", linewidth=2.5, label="NPV")
    ax_main.fill_between(thresholds, ppv, alpha=0.08, color="#E74C3C")

    # Add sensitivity and specificity as thin lines
    ax_main.plot(thresholds, sensitivity, color="#27AE60", linewidth=1.2, linestyle="--", alpha=0.7, label="Sensitivity")
    ax_main.plot(thresholds, specificity, color="#8E44AD", linewidth=1.2, linestyle="--", alpha=0.7, label="Specificity")

    ax_main.set_ylabel("Metric value", fontsize=11)
    ax_main.set_ylim(0, 1.02)
    ax_main.legend(fontsize=10, ncol=2, loc="center right")
    ax_main.grid(True, alpha=0.3)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Histogram of predicted probabilities
    ax_hist.hist(y_prob[y_true == 0], bins=100, range=(0, 1), alpha=0.6, color="#2980B9", label="Negative", density=True)
    ax_hist.hist(y_prob[y_true == 1], bins=100, range=(0, 1), alpha=0.6, color="#E74C3C", label="Positive", density=True)
    ax_hist.set_xlabel("Decision threshold / Predicted probability", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.set_xlim(0, 1)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Load predictions
    with open(
        "/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/"
        "best_xgb_final_model/test_predictions.pkl",
        "rb",
    ) as f:
        y_test, y_prob = pickle.load(f)

    # Compute metrics at many thresholds
    thresholds = np.linspace(0.001, 0.999, 1000)
    ppv, npv, sensitivity, specificity = compute_ppv_npv(y_test, y_prob, thresholds)

    output_dir = (
        "/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/"
        "best_xgb_final_model/ppv_npv_plots"
    )
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_ppv_npv(y_test, y_prob, thresholds, ppv, npv, sensitivity, specificity,
                       smoothing_window=51)

    for fmt in ["png", "tiff"]:
        path = os.path.join(output_dir, f"ppv_npv_threshold_plot.{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

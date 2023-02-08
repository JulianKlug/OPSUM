from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_calibration_curve(y_true, y_prob, n_bins=5, ax=None, hist=True, normalize=False, title=None):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, normalize=normalize)
    if ax is None:
        ax = plt.gca()
    if hist:
        ax.hist(y_prob, weights=np.ones_like(y_prob) / len(y_prob), alpha=.4,
               bins=np.maximum(10, n_bins))
    ax.plot([0, 1], [0, 1], ':', c='k')
    curve = ax.plot(prob_pred, prob_true, marker="o")

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("fraction of positive samples")

    ax.set(aspect='equal')

    if title is not None:
        ax.set_title(title)

    return curve
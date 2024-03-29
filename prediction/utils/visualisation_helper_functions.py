import numpy as np
from colormath.color_objects import sRGBColor
from colormath.color_conversions import convert_color
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
import matplotlib.text as mtext


def reverse_normalisation_for_subj(norm_subj_df, normalisation_parameters_df):
    subj_df = norm_subj_df.copy()
    for variable in normalisation_parameters_df.variable.unique():
        if variable not in subj_df.columns:
            continue

        temp = subj_df[variable].copy()
        std = normalisation_parameters_df[normalisation_parameters_df.variable == variable].original_std.iloc[0]
        mean = normalisation_parameters_df[normalisation_parameters_df.variable == variable].original_mean.iloc[0]
        temp = (temp * std) + mean
        subj_df[variable] = temp

    return subj_df


def create_palette(start_rgb, end_rgb, n, colorspace, extrapolation_length):
    # convert start and end to a point in the given colorspace
    start = np.array(convert_color(start_rgb, colorspace, observer=2).get_value_tuple())
    mid = np.array(convert_color(end_rgb, colorspace, observer=2).get_value_tuple())

    # extrapolate the end point
    end = start + extrapolation_length * (mid - start)

    # create a set of n points along start to end
    points = list(zip(*[np.linspace(start[i], end[i], n) for i in range(3)]))

    # create a color for each point and convert back to rgb
    rgb = [convert_color(colorspace(*point), sRGBColor).get_value_tuple() for point in points]

    # rgb_colors = np.maximum(np.minimum(rgb, [1, 1, 1]), [0, 0, 0])

    rgb_colors = []
    for color in rgb:
        c = list(color)
        for i in range(3):
            if c[i] > 1:
                c[i] = 2 - c[i]
            if c[i] < 0:
                c[i] *= -1
        rgb_colors.append(c)

    # finally convert rgb colors back to hex
    return [sRGBColor(*color).get_rgb_hex() for color in rgb_colors]

def hex_to_rgb_color(hex):
    return sRGBColor(*[int(hex[i + 1:i + 3], 16) for i in (0, 2 ,4)], is_upscaled=True)


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

def density_jitter(values, width=1.0, cluster_factor=1.0):
    """
    Add jitter to a 1D array of values, using a kernel density estimate
    """
    inds = np.arange(len(values))
    np.random.shuffle(inds)
    values = values[inds]
    N = len(values)
    nbins = 100
    quant = np.round(nbins * (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8))
    inds = np.argsort(quant + np.random.randn(N) * 1e-6)
    layer = 0
    last_bin = -1
    ys = np.zeros(N)
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = cluster_factor * (np.ceil(layer / 2) * ((layer % 2) * 2 - 1))
        layer += 1
        last_bin = quant[ind]
    ys *= 0.9 * (width / np.max(ys + 1))

    return ys


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
        handlebox.add_artist(title)
        return title

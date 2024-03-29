import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from prediction.utils.utils import ensure_tensor


def precision(y_true, y_pred):
    # Ensure that y_true and y_pred are tensors (replaces K.constant, which fails if tensor is passed)
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def specificity(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels
    Returns:
    Specificity score
    """
    # Ensure that y_true and y_pred are tensors (replaces K.constant, which fails if tensor is passed)
    neg_y_true = ensure_tensor(1 - y_true)
    neg_y_pred = ensure_tensor(1 - y_pred)
    y_pred = ensure_tensor(y_pred)

    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


def recall(y_true, y_pred):
    # Ensure that y_true and y_pred are tensors (replaces K.constant, which fails if tensor is passed)
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def matthews(y_true, y_pred):
    # Ensure that y_true and y_pred are tensors (replaces K.constant, which fails if tensor is passed)
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())


def plot_roc_curve(y_true, y_pred, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(7.5, 5))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    # adjust figure size

    return fig


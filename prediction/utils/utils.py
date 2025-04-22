import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import torch
import torch.nn as nn

# define function for balanced training
def generate_balanced_arrays(X_train, y_train):
    while True:
        initial_positive = np.where(y_train == 1)[0].tolist()
        initial_negative = np.where(y_train == 0)[0].tolist()

        if len(initial_positive) < len(initial_negative):
            # if there are more negative samples than positive samples
            positive = initial_positive
            negative = np.random.choice(initial_negative, len(initial_positive), replace=False).tolist()
        else:
            # If there are more positive samples than negative samples, we need to downsample the positive samples
            positive = np.random.choice(initial_positive, len(initial_negative), replace=False).tolist()
            negative = initial_negative

        balance = np.concatenate((positive, negative), axis=0)
        np.random.shuffle(balance)
        input_ = X_train[balance]
        target = y_train[balance]
        yield input_, target


def filter_consecutive_numbers(lst):
    a = np.array(list(lst)).astype(int)
    if len(lst) < 2:
        return a
    consecutive_mask = np.concatenate(([False], (np.abs(a[1:] - a[:-1]) == 1)))
    result = a[np.logical_not(consecutive_mask)]
    if len(lst) == 2:
        return result
    next_to_consecutive_mask = np.concatenate(([False], (np.abs(result[1:] - result[:-1]) == 2)))
    result = result[np.logical_not(next_to_consecutive_mask)]
    return np.array(result)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def moving_time_average(a, n=3):
    """
    This function calculates the moving average over the last n elements of the array a.
    """
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n


def aggregate_features_over_time(features, labels, moving_average=False, n=3):
    """
    This function aggregates the features over time. Instead of having one row per case_admission_id and one column per feature over time,
    we have one row per case_admission_id and one column per feature aggregated over time (mean, min, max) for each timestep.
    The timesteps are then flattened along with the samples.

    :param features: a numpy array of shape (n_samples, n_time_steps, n_features)
    :param labels: a numpy array of shape (n_samples, 1)
    :param moving_average: if True, the moving average over the last n time steps is calculated
    :param n: the number of time steps for the moving average
    """
    avg_features = np.cumsum(features, 1) / (np.arange(1, features.shape[1] + 1)[None, :, None])
    if moving_average:
        avg_features = np.append(avg_features[:, :n - 1], moving_time_average(features, n), axis=1)

    min_features = np.minimum.accumulate(features, 1)
    max_features = np.maximum.accumulate(features, 1)
    all_features = np.concatenate([features, avg_features, min_features, max_features], 2)
    all_features = all_features.reshape(-1, all_features.shape[-1])

    labels = labels[:, None].repeat(72, 1).ravel()

    return all_features, labels

def flatten(l):
    return [item for sublist in l for item in sublist]

def check_data(data):
    """
    Check if data contains nan or inf
    """
    if type(data) == np.ndarray:
        if np.isnan(data).any() or np.isinf(data).any():
            sys.exit('Data is corrupted!')
            return False
    elif type(data) == pd.DataFrame:
        if data.isnull().values.any() or data.isin([np.inf, -np.inf]).values.any():
            sys.exit('Data is corrupted!')
            return False
    else:
        return True


def ensure_dir(dirname: Path) -> None:
    """
    Create directory only if it does not exist yet.
    Throw an error otherwise.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def flatten(t):
    return [item for sublist in t for item in sublist]


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def ensure_tensor(x):
    if not isinstance(x, tf.Tensor):
        x = K.constant(x)
    return x

def calc_iou(a, b):

    a=a.type(torch.cuda.DoubleTensor)
    b=b.type(torch.cuda.DoubleTensor)

    area = (b[:, 2] - b[:, 0]+1) * (b[:, 3] - b[:, 1]+1)

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])+1
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])+1

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]+1) * (a[:, 3] - a[:, 1]+1), dim=1) + area - iw * ih

    #ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Implements Focal Loss.

        Args:
            alpha (float, optional): Weighting factor for the rare class (outliers). Default: 0.25
            gamma (float, optional): Focusing parameter to reduce loss for easy examples. Default: 2.0
            reduction (str, optional): 'mean' (default), 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Model output logits before sigmoid activation.
            targets (Tensor): Binary ground truth labels.

        Returns:
            Tensor: Computed focal loss.
        """
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        targets = targets.float()  # Ensure targets are float for calculations

        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t for focal loss
        focal_weight = (1 - pt) ** self.gamma  # Compute focal weight

        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss  # No reduction


class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, classifications, regressions, anchors, annotations):

        batch_size = classifications.shape[0]
        regression_losses = []

        regression_grads = torch.zeros(regressions.shape).cuda()
        p_num = torch.zeros(1).cuda()
        labels_b = []

        anchor = anchors[0, :, :].type(torch.cuda.FloatTensor)

        anchor_widths = anchor[:, 2] - anchor[:, 0] + 1.0
        anchor_heights = anchor[:, 3] - anchor[:, 1] + 1.0
        anchor_ctr_x = anchor[:, 0] + 0.5 * (anchor_widths - 1.0)
        anchor_ctr_y = anchor[:, 1] + 0.5 * (anchor_heights - 1.0)

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                labels_b.append(torch.zeros(classification.shape).cuda())
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            ######
            gt_IoU_max, gt_IoU_argmax = torch.max(IoU, dim=0)
            gt_IoU_argmax = torch.where(IoU == gt_IoU_max)[0]
            positive_indices = torch.ge(torch.zeros(IoU_max.shape).cuda(), 1)
            positive_indices[gt_IoU_argmax.long()] = True
            ######

            positive_indices = positive_indices | torch.ge(IoU_max, 0.5)
            negative_indices = torch.lt(IoU_max, 0.4)

            p_num += positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[negative_indices, :] = 0
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            labels_b.append(targets)

            # compute the loss for regression
            if positive_indices.sum() > 0:

                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0] + 1.0
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1] + 1.0
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * (gt_widths - 1.0)
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * (gt_heights - 1.0)

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets2 = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets2 = targets2.t()

                targets2 = targets2 / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                # negative_indices = ~ positive_indices

                regression_diff = regression[positive_indices, :] - targets2
                regression_diff_abs = torch.abs(regression_diff)

                regression_loss = torch.where(
                    torch.le(regression_diff_abs, 1.0 / 1.0),
                    0.5 * 1.0 * torch.pow(regression_diff_abs, 2),
                    regression_diff_abs - 0.5 / 1.0
                )
                regression_losses.append(regression_loss.sum())

                regression_grad = torch.where(
                    torch.le(regression_diff_abs, 1.0 / 1.0),
                    1.0 * regression_diff,
                    torch.sign(regression_diff))
                regression_grads[j, positive_indices, :] = regression_grad

            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        p_num = torch.clamp(p_num, min=1)
        regression_grads /= (4 * p_num)

        ########################AP-LOSS##########################
        labels_b = torch.stack(labels_b)
        classification_grads, classification_losses = AP_loss(classifications, labels_b)
        #########################################################

        ctx.save_for_backward(classification_grads, regression_grads)
        return classification_losses, torch.stack(regression_losses).sum(dim=0, keepdim=True) / p_num

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, g2 = ctx.saved_tensors
        return g1 * out_grad1, g2 * out_grad2, None, None


def AP_loss(logits, targets):
    delta = 1.0

    grad = torch.zeros(logits.shape).cuda()
    metric = torch.zeros(1).cuda()

    if torch.max(targets) <= 0:
        return grad, metric

    labels_p = (targets == 1)
    fg_logits = logits[labels_p]
    threshold_logit = torch.min(fg_logits) - delta

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n = ((targets == 0) & (logits >= threshold_logit))
    valid_bg_logits = logits[valid_labels_n]
    valid_bg_grad = torch.zeros(len(valid_bg_logits)).cuda()
    ########

    fg_num = len(fg_logits)
    prec = torch.zeros(fg_num).cuda()
    order = torch.argsort(fg_logits)
    max_prec = 0

    for ii in order:
        tmp1 = fg_logits - fg_logits[ii]
        tmp1 = torch.clamp(tmp1 / (2 * delta) + 0.5, min=0, max=1)
        tmp2 = valid_bg_logits - fg_logits[ii]
        tmp2 = torch.clamp(tmp2 / (2 * delta) + 0.5, min=0, max=1)
        a = torch.sum(tmp1) + 0.5
        b = torch.sum(tmp2)
        tmp2 /= (a + b)
        current_prec = a / (a + b)
        if (max_prec <= current_prec):
            max_prec = current_prec
        else:
            tmp2 *= ((1 - max_prec) / (1 - current_prec))
        valid_bg_grad += tmp2
        prec[ii] = max_prec

    grad[valid_labels_n] = valid_bg_grad
    grad[labels_p] = -(1 - prec)

    fg_num = max(fg_num, 1)

    grad /= (fg_num)

    metric = torch.sum(prec, dim=0, keepdim=True) / fg_num

    return grad, 1 - metric


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_idx: int, weight_value: float, vector_length: int):
        """
        Args:
            weight_idx (int): Index of the variable to weight more.
            weight_value (float): Weight multiplier for the selected index.
            vector_length (int): Length of the vector (e.g., 84).
        """
        super().__init__()
        weights = torch.ones(vector_length)
        weights[weight_idx] = weight_value
        self.register_buffer('weights', weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten last dimension (assuming shape [batch_size, ..., vector_length])
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        loss = ((pred_flat - target_flat) ** 2) * self.weights
        return loss.mean()
    

class WeightedCosineSimilarity(nn.Module):
    def __init__(self, weight_idx: int, weight_value: float, vector_length: int, reduction: str = 'mean'):
        """
        Args:
            weight_idx (int): Index of the variable to weight more.
            weight_value (float): Multiplier for the selected variable.
            vector_length (int): Length of the vector (e.g., 84).
            reduction (str): 'mean' or 'none'
        """
        super().__init__()
        weights = torch.ones(vector_length)
        weights[weight_idx] = weight_value
        self.register_buffer('weights', weights)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y assumed to be shape [batch_size, vector_length]
        x = x * self.weights
        y = y * self.weights

        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)

        cos_sim = (x_norm * y_norm).sum(dim=-1)  # shape: [batch_size]
        if self.reduction == 'mean':
            return cos_sim.mean()
        else:
            return cos_sim  # shape: [batch_size]
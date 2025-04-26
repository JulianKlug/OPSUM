import torch
import torch.nn as nn
from torchsort import soft_rank
import torch.nn.functional as F


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


class SoftAPLoss(nn.Module):
    def __init__(self, regularization: str = 'l2', tau: float = 1.0):
        """
        Args:
            regularization: one of {'l2','kl'} passed to soft_rank
            tau: temperature for the soft ranking (lower = smoother)
        """
        super().__init__()
        self.reg = regularization
        self.tau = tau

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores:  shape (batch_size,) — raw model outputs (higher = more positive)
            targets: shape (batch_size,) — binary {0,1} labels
        Returns:
            A scalar loss = 1 − SoftAP
        """
        # sanity checks
        if scores.dim() != 1 or targets.dim() != 1 or scores.shape != targets.shape:
            raise ValueError("scores and targets must be 1D tensors of the same shape")
        y = targets.float()
        N = scores.size(0)

        # 1) compute soft ranks (shape: [N])
        ranks = soft_rank(
            scores.unsqueeze(0),
            regularization=self.reg,
            temperature=self.tau
        ).squeeze(0)

        # 2) build pairwise indicator H_ij ≈ 1[r_j ≤ r_i]
        diff = ranks.unsqueeze(1) - ranks.unsqueeze(0)   # (N,N)
        H = torch.sigmoid(diff * self.tau)               # (N,N)

        # 3) soft true-positives above each i
        tp = (H * y.unsqueeze(0)).sum(dim=1)             # (N,)

        # 4) precision at each i, averaged only over positives
        precision_i = tp / ranks
        P = y.sum().clamp(min=1.0)                       # avoid div by zero
        soft_ap = (precision_i * y).sum() / P

        # 5) loss
        return 1.0 - soft_ap



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
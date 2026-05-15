import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric


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
    def __init__(self, tau: float = 1.0, eps: float = 1e-6):
        """
        A soft?~@~PAP loss that does not rely on torchsort.

        Args:
            tau: temperature for the sigmoid approximation (smaller = smoother)
            eps: small constant for numerical stability
        """
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.tau = tau
        self.eps = eps

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: 1D float tensor of model outputs (higher ?~F~R more positive), shape (N,)
            targets: 1D binary tensor of labels {0,1}, shape (N,)

        Returns:
            scalar loss = 1 ?~H~R (soft AP)
        """
        if scores.dim() != 1 or targets.dim() != 1 or scores.shape != targets.shape:
            raise ValueError("scores and targets must be 1D tensors of same shape")

        y = targets.float()
        N = scores.size(0)

        # 1) pairwise differences
        diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (N,N)

        # 2) soft comparison matrix Hij ?~I~H 1{score_j ?~I? score_i}
        H = torch.sigmoid((-diff) / self.tau)             # (N,N)

        # instead of in-place diag assignment, build a new matrix
        eye = torch.eye(N, device=scores.device, dtype=scores.dtype)
        H = H * (1.0 - eye) + eye                         # now H[ii]=1 but no in-place

        # 3) soft true-positives above each threshold
        tp = (H * y.unsqueeze(0)).sum(dim=1)              # (N,)

        # 4) soft "predicted positives" above each threshold
        denom = H.sum(dim=1).clamp(min=self.eps)          # (N,)

        # 5) precision@i and average over true positives
        precision = tp / denom                            # (N,)
        P = y.sum().clamp(min=self.eps)
        soft_ap = (precision * y).sum() / P               # scalar

        # 6) final loss
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
    

class WeightedCosineSimilarity(Metric):
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
        self.add_state("sum_cos_sim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x: torch.Tensor, y: torch.Tensor):
        # x, y assumed to be shape [batch_size, vector_length]
        x = x * self.weights
        y = y * self.weights

        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)

        cos_sim = (x_norm * y_norm).sum(dim=-1)  # shape: [batch_size]
        
        if self.reduction == 'mean':
            self.sum_cos_sim += cos_sim.sum()
            self.total += x.shape[0]
        else:
            # Store individual similarities
            self.sum_cos_sim = cos_sim
            self.total = torch.tensor(x.shape[0], device=x.device)

    def compute(self):
        if self.reduction == 'mean':
            return self.sum_cos_sim / self.total if self.total > 0 else torch.tensor(0.0)
        else:
            return self.sum_cos_sim  # shape: [batch_size]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # For backward compatibility
        x = x * self.weights
        y = y * self.weights

        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)

        cos_sim = (x_norm * y_norm).sum(dim=-1)  # shape: [batch_size]
        if self.reduction == 'mean':
            return cos_sim.mean()
        else:
            return cos_sim  # shape: [batch_size]
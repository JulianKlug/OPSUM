"""
MIRA finetuning model for binary classification of early neurological deterioration.

Adds a classification head on top of MIRA's transformer backbone.
The backbone can be frozen or partially unfrozen for finetuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from mira_inference import load_mira_model


class MIRAClassifier(nn.Module):
    """
    Binary classifier built on top of MIRA's pretrained backbone.

    Architecture:
        MIRA backbone -> pooled hidden state -> classification head -> sigmoid

    The model takes a single-channel time series (e.g., max_NIHSS) and predicts
    the probability of early neurological deterioration.
    """

    def __init__(
        self,
        model_name: str = "MIRA-Mode/MIRA",
        device: str = "cuda",
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.device_str = device

        # Load pretrained MIRA backbone
        self.backbone = load_mira_model(model_name, device='cpu', disable_ode=True)
        self.backbone.train()
        self.hidden_size = self.backbone.config.hidden_size  # 4096

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Optionally unfreeze last N transformer layers
            if unfreeze_last_n_layers > 0:
                total_layers = self.backbone.config.num_hidden_layers
                for i in range(total_layers - unfreeze_last_n_layers, total_layers):
                    for param in self.backbone.model.layers[i].parameters():
                        param.requires_grad = True
                # Also unfreeze the final norm
                for param in self.backbone.model.norm.parameters():
                    param.requires_grad = True

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.FloatTensor,
        time_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len, 1] - historical values
            time_values: [batch, seq_len] - timestamps
            attention_mask: [batch, seq_len] - optional attention mask

        Returns:
            logits: [batch, 1] - classification logits (pre-sigmoid)
        """
        # Get hidden states from MIRA backbone
        outputs = self.backbone.model(
            input_ids=input_ids,
            time_values=time_values,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Pool: use last token hidden state
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        pooled = hidden_states[:, -1, :]  # [batch, hidden_size]

        # Classify
        logits = self.classifier(pooled)  # [batch, 1]
        return logits

    def get_trainable_params(self):
        """Return count of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

"""
PyTorch Dataset for MIRA finetuning on early neurological deterioration prediction.

Transforms OPSUM data splits into per-timestep samples suitable for
training a binary classifier on top of MIRA.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from mira_data_loader import (
    get_feature_indices,
    get_scaler,
    apply_scaler,
    get_validation_patient_ids,
)
from mira_inference import normalize_sequence


class MIRAFinetuneDataset(Dataset):
    """
    Dataset that produces (input_ids, time_values, label) tuples.

    For each case and each timestep, the sample consists of:
    - input_ids: max_NIHSS time series up to (and including) that timestep
    - time_values: hourly timestamps [0, 1, ..., timestep]
    - label: binary label for neurological deterioration at timestep + forecast_horizon
    """

    def __init__(
        self,
        X: np.ndarray,
        X_for_scaler: np.ndarray,
        outcome_df: pd.DataFrame,
        n_time_steps: int = 72,
        forecast_horizon: int = 6,
        scaler: StandardScaler = None,
    ):
        """
        Args:
            X: Data array [n_cases, n_time_steps, n_features, n_dims]
            X_for_scaler: Data to fit scaler on (typically training data)
            outcome_df: Outcome labels DataFrame
            n_time_steps: Number of timesteps to use
            forecast_horizon: How many hours ahead the label refers to
            scaler: Pre-fitted scaler (if None, will be fit on X_for_scaler)
        """
        self.n_time_steps = n_time_steps
        self.forecast_horizon = forecast_horizon

        # Get feature indices
        min_nihss_idx, max_nihss_idx = get_feature_indices(X)
        self.max_nihss_idx = max_nihss_idx

        # Fit/apply scaler
        if scaler is None:
            scaler = get_scaler(X_for_scaler)
        self.scaler = scaler
        X_scaled = apply_scaler(X, scaler)

        # Extract max_NIHSS time series
        self.max_nihss = X_scaled[:, :, max_nihss_idx].astype(np.float32)  # [n_cases, n_time_steps]

        # Get patient IDs
        patient_cids = get_validation_patient_ids(X)

        # Build sample list: (case_idx, timestep, label)
        self.samples = []
        for ts in range(n_time_steps):
            evaluated_ts = ts + forecast_horizon
            outcome_at_ts = outcome_df[
                outcome_df['relative_sample_date_hourly_cat'] == evaluated_ts
            ]
            labels = np.isin(
                patient_cids, outcome_at_ts['case_admission_id'].values
            ).astype(np.float32)

            for case_idx in range(len(X)):
                self.samples.append((case_idx, ts, labels[case_idx]))

        # Compute class weights for imbalanced data
        labels_array = np.array([s[2] for s in self.samples])
        n_pos = labels_array.sum()
        n_neg = len(labels_array) - n_pos
        if n_pos > 0:
            self.pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        else:
            self.pos_weight = torch.tensor([1.0], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, ts, label = self.samples[idx]

        # Extract max_NIHSS up to current timestep
        values = self.max_nihss[case_idx, :ts + 1]  # [ts+1]
        input_ids = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)  # [ts+1, 1]
        time_values = torch.arange(ts + 1, dtype=torch.float32)  # [ts+1]
        label = torch.tensor([label], dtype=torch.float32)

        return input_ids, time_values, label


def collate_fn(batch):
    """
    Custom collate that pads variable-length sequences.

    Returns:
        input_ids: [batch, max_seq_len, 1] (zero-padded)
        time_values: [batch, max_seq_len] (zero-padded)
        attention_mask: [batch, max_seq_len] (1 for real, 0 for padding)
        labels: [batch, 1]
    """
    input_ids_list, time_values_list, labels_list = zip(*batch)

    max_len = max(x.shape[0] for x in input_ids_list)

    padded_inputs = []
    padded_times = []
    masks = []

    for inp, tv in zip(input_ids_list, time_values_list):
        seq_len = inp.shape[0]
        pad_len = max_len - seq_len

        padded_inp = F.pad(inp, (0, 0, 0, pad_len), value=0.0)
        padded_tv = F.pad(tv, (0, pad_len), value=0.0)
        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])

        padded_inputs.append(padded_inp)
        padded_times.append(padded_tv)
        masks.append(mask)

    return (
        torch.stack(padded_inputs),    # [B, L, 1]
        torch.stack(padded_times),     # [B, L]
        torch.stack(masks),            # [B, L]
        torch.stack(labels_list),      # [B, 1]
    )

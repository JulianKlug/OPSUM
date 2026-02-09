"""
MIRA inference module for zero-shot time series forecasting.

This module provides functions to load the MIRA model and forecast
future values of NIHSS features for early neurological deterioration prediction.
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm

# Add MIRA to path
MIRA_PATH = os.path.join(os.path.dirname(__file__), 'MIRA')
if MIRA_PATH not in sys.path:
    sys.path.insert(0, MIRA_PATH)

from MIRA.models.modeling_mira import MIRAForPrediction
from MIRA.models.configuration_mira import MIRAConfig


def load_mira_model(model_name: str = "MIRA-Mode/MIRA", device: str = "cuda", disable_ode: bool = True) -> MIRAForPrediction:
    """
    Load pretrained MIRA model from HuggingFace.

    Args:
        model_name: HuggingFace model name or local path
        device: Device to load model on ('cuda' or 'cpu')
        disable_ode: Whether to disable ODE extrapolation (default True to avoid numerical issues)

    Returns:
        Loaded MIRAForPrediction model
    """
    if disable_ode:
        # Load config first and disable ODE before model initialization
        config = MIRAConfig.from_pretrained(model_name)
        config.use_terminal_ode = False
        model = MIRAForPrediction.from_pretrained(model_name, config=config)
        # Also set model attributes to ensure ODE is disabled
        model.use_terminal_ode = False
        model.ode_extrapolation_block = None
        # Verify ODE is disabled
        print(f"  ODE disabled: use_terminal_ode={model.use_terminal_ode}, ode_block={model.ode_extrapolation_block}")
    else:
        model = MIRAForPrediction.from_pretrained(model_name)

    model = model.to(device)
    model.eval()
    return model


def normalize_sequence(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize time series values per sequence (mean=0, std=1).

    Args:
        values: Input values of shape [batch, seq_len] or [batch, seq_len, 1]

    Returns:
        Tuple of (normalized_values, mean, std)
    """
    if values.dim() == 3:
        values = values.squeeze(-1)

    mean = values.mean(dim=1, keepdim=True)

    # Handle single-element sequences (std would be 0 or undefined)
    if values.shape[1] == 1:
        # For single elements, use std=1 to avoid division issues
        std = torch.ones_like(mean)
    else:
        std = values.std(dim=1, keepdim=True)
        # Replace zero std with 1 to avoid division by zero
        std = torch.where(std < 1e-8, torch.ones_like(std), std)

    std = std + 1e-8  # Small epsilon for numerical stability

    normalized = (values - mean) / std

    return normalized, mean, std


def denormalize_sequence(normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize time series values.

    Args:
        normalized: Normalized values
        mean: Original mean
        std: Original std

    Returns:
        Denormalized values
    """
    if normalized.dim() == 3:
        normalized = normalized.squeeze(-1)
    if mean.dim() == 1:
        mean = mean.unsqueeze(1)
    if std.dim() == 1:
        std = std.unsqueeze(1)

    return normalized * std + mean


def forecast_single_step(
    model: MIRAForPrediction,
    input_ids: torch.Tensor,
    time_values: torch.Tensor,
    next_time: torch.Tensor
) -> torch.Tensor:
    """
    Forecast a single step ahead.

    Args:
        model: MIRA model
        input_ids: Historical values [batch, seq_len, 1]
        time_values: Historical timestamps [batch, seq_len]
        next_time: Next timestamp to predict [batch, 1]

    Returns:
        Predicted value [batch, 1]
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            time_values=time_values,
            next_target_time_values=next_time.squeeze(-1) if next_time.dim() == 3 else next_time,
            return_dict=True,
        )

    # Get the last prediction
    prediction = outputs.logits[:, -1, :]  # [batch, 1]
    return prediction


def forecast_multistep(
    model: MIRAForPrediction,
    input_ids: torch.Tensor,
    time_values: torch.Tensor,
    n_steps: int = 6,
    time_step: float = 1.0
) -> torch.Tensor:
    """
    Forecast multiple steps ahead autoregressively.

    MIRA internally normalizes input, so we do the same normalization
    and denormalization on our side.

    Args:
        model: MIRA model
        input_ids: Historical values [batch, seq_len, 1] (already normalized by MIRA internally)
        time_values: Historical timestamps [batch, seq_len]
        n_steps: Number of steps to forecast
        time_step: Time increment between steps (default 1.0 for hourly)

    Returns:
        Forecasted values [batch, n_steps]
    """
    device = next(model.parameters()).device
    batch_size = input_ids.shape[0]

    # Normalize input values per sequence
    input_values = input_ids.squeeze(-1)  # [batch, seq_len]
    norm_input, mean, std = normalize_sequence(input_values)
    norm_input = norm_input.unsqueeze(-1).to(device)  # [batch, seq_len, 1]

    current_input = norm_input.clone()
    current_times = time_values.clone().to(device)

    predictions = []

    for step in range(n_steps):
        # Calculate next time
        last_time = current_times[:, -1:]  # [batch, 1]
        next_time = last_time + time_step  # [batch, 1]

        # Get prediction for next step
        with torch.no_grad():
            outputs = model(
                input_ids=current_input,
                time_values=current_times,
                next_target_time_values=None,  # Don't use ODE (disabled in model)
                return_dict=True,
            )

        # Get the predicted value (still normalized)
        next_pred_norm = outputs.logits[:, -1, :]  # [batch, 1]
        predictions.append(next_pred_norm)

        # Update inputs for next iteration
        current_input = torch.cat([current_input, next_pred_norm.unsqueeze(-1)], dim=1)
        current_times = torch.cat([current_times, next_time], dim=1)

    # Stack predictions [batch, n_steps]
    predictions = torch.cat(predictions, dim=1)

    # Denormalize predictions
    predictions = denormalize_sequence(predictions, mean.to(device), std.to(device))

    return predictions


def forecast_nihss_at_timestep(
    model: MIRAForPrediction,
    X_val_scaled: np.ndarray,
    max_nihss_idx: int,
    current_timestep: int,
    n_forecast_steps: int = 6,
    batch_size: int = 64
) -> np.ndarray:
    """
    Forecast max_NIHSS values for all validation samples at a given timestep.

    Args:
        model: MIRA model
        X_val_scaled: Scaled validation data [n_cases, n_time_steps, n_features]
        max_nihss_idx: Index of max_NIHSS feature
        current_timestep: Current timestep to forecast from
        n_forecast_steps: Number of steps to forecast ahead
        batch_size: Batch size for inference

    Returns:
        Forecasted max_NIHSS values [n_cases, n_forecast_steps]
    """
    device = next(model.parameters()).device
    n_cases = X_val_scaled.shape[0]

    # Extract max_NIHSS time series up to current timestep
    max_nihss_history = X_val_scaled[:, :current_timestep + 1, max_nihss_idx]  # [n_cases, ts+1]

    all_forecasts = []

    for start_idx in range(0, n_cases, batch_size):
        end_idx = min(start_idx + batch_size, n_cases)
        batch_history = max_nihss_history[start_idx:end_idx]

        # Prepare MIRA input
        input_ids = torch.tensor(batch_history, dtype=torch.float32).unsqueeze(-1)  # [batch, ts+1, 1]
        seq_len = input_ids.shape[1]
        time_values = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).expand(input_ids.shape[0], -1)

        # Forecast
        forecasts = forecast_multistep(
            model,
            input_ids.to(device),
            time_values.to(device),
            n_steps=n_forecast_steps
        )

        all_forecasts.append(forecasts.cpu().numpy())

    return np.concatenate(all_forecasts, axis=0)


def forecast_all_timesteps(
    model: MIRAForPrediction,
    X_val_scaled: np.ndarray,
    max_nihss_idx: int,
    n_time_steps: int = 72,
    n_forecast_steps: int = 6,
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
    Forecast max_NIHSS values for all validation samples across all timesteps.

    Args:
        model: MIRA model
        X_val_scaled: Scaled validation data [n_cases, n_time_steps, n_features]
        max_nihss_idx: Index of max_NIHSS feature
        n_time_steps: Total number of timesteps
        n_forecast_steps: Number of steps to forecast ahead
        batch_size: Batch size for inference
        show_progress: Whether to show progress bar

    Returns:
        List of forecasted values, one array per timestep
        Each array has shape [n_cases, n_forecast_steps]
    """
    all_timestep_forecasts = []

    timestep_range = range(n_time_steps)
    if show_progress:
        timestep_range = tqdm(timestep_range, desc="Forecasting timesteps")

    for ts in timestep_range:
        forecasts = forecast_nihss_at_timestep(
            model,
            X_val_scaled,
            max_nihss_idx,
            current_timestep=ts,
            n_forecast_steps=n_forecast_steps,
            batch_size=batch_size
        )
        all_timestep_forecasts.append(forecasts)

    return all_timestep_forecasts

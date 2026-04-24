# XGB Feature Aggregation Methods

**Source:** `prediction/utils/utils.py` — `aggregate_features_over_time()`

## Overview

The XGBoost model operates on tabular data, not raw time series. The input time series of shape `(n_patients, 72 timesteps, n_features)` is transformed into a flat feature matrix where each patient-timestep pair becomes one row, and each column represents either the raw value of a feature or a derived statistic computed up to that timestep.

At each timestep `t`, the model sees the raw feature values **plus** a set of summary statistics computed from all data available up to `t`. This gives the model both the current state and the historical context.

After aggregation, the feature matrix is flattened to shape `(n_patients * 72, n_total_features)` and labels are repeated per timestep.

---

## Base Features (always enabled)

### 1. Raw values

The original feature values at each timestep, unchanged.

- Shape per timestep: `(n_features,)`
- `features[:, t, :]` = raw value at time `t`

### 2. Cumulative mean

Running average of each feature from timestep 0 up to the current timestep `t`.

```
avg(t) = (1 / (t+1)) * sum(features[0..t])
```

Computed efficiently via `np.cumsum(features, axis=1) / counts`.

Captures the overall trend of a feature across the patient's stay. Useful for distinguishing patients whose values have been consistently high vs. transiently elevated.

### 3. Cumulative minimum

Running minimum of each feature from timestep 0 to `t`.

```
min(t) = min(features[0], features[1], ..., features[t])
```

Computed via `np.minimum.accumulate(features, axis=1)`.

Captures the worst (or best, depending on the feature) value observed so far. Clinically relevant for features like blood pressure or GCS where the nadir may be prognostic.

### 4. Cumulative maximum

Running maximum of each feature from timestep 0 to `t`.

```
max(t) = max(features[0], features[1], ..., features[t])
```

Computed via `np.maximum.accumulate(features, axis=1)`.

Captures the peak value observed so far. Useful for features like temperature or heart rate where spikes may indicate clinical events.

### 5. Cumulative standard deviation

Running standard deviation from timestep 0 to `t`.

```
std(t) = sqrt(mean(x^2[0..t]) - mean(x[0..t])^2)
```

Computed via the identity `Var(X) = E[X^2] - E[X]^2`, using cumulative sums of `features` and `features^2`. Clamped to non-negative before taking the square root to avoid numerical issues.

Captures clinical instability or variability. A patient with high cumulative std in blood pressure, for example, exhibits more haemodynamic instability.

### 6. Rate of change (first-order differences)

The change in each feature between consecutive timesteps.

```
diff(t) = features[t] - features[t-1]    (for t >= 1)
diff(0) = 0                               (zero-padded)
```

Captures acute changes, i.e., whether a feature is currently rising or falling and by how much. A sudden drop in consciousness level or spike in heart rate appears as a large absolute difference.

### 7. Timestep index

A single scalar feature representing the normalised position in time.

```
timestep(t) = t / (n_timesteps - 1)
```

Ranges from 0.0 (first timestep) to 1.0 (last timestep). Gives the model awareness of temporal position, allowing it to learn that the same feature value may have different prognostic significance early vs. late in the stay.

---

## Optional: Lag Features (`add_lag_features=True`)

### 8. Lag-2 (value at t-2)

The raw feature value from 2 timesteps ago.

```
lag2(t) = features[t-2]    (for t >= 2)
lag2(t) = 0                (for t < 2, zero-padded)
```

### 9. Lag-3 (value at t-3)

The raw feature value from 3 timesteps ago.

```
lag3(t) = features[t-3]    (for t >= 3)
lag3(t) = 0                (for t < 3, zero-padded)
```

### Rationale for lag features

Lag features give the model direct access to recent historical values without having to infer them from summary statistics. Combined with the raw value at `t`, the model sees a 3-point trajectory (t-3, t-2, t) which can capture short-term patterns like acceleration or deceleration of clinical decline. The lag-1 value (`t-1`) is implicitly available through the combination of the raw value and the diff feature.

---

## Optional: Rolling Window Features (`add_rolling_features=True`)

These features are computed over a sliding window of the last `w` timesteps (default `w=6`, i.e., 6 hours). For timesteps where fewer than `w` values are available (`t < w`), all available values are used.

Rolling features complement the cumulative statistics by focusing on **recent** dynamics. As time progresses, cumulative statistics become increasingly diluted by early values, making them insensitive to recent changes. Rolling features maintain a fixed-width focus on the most recent period.

### 10. Rolling mean

Average of feature values over the last `w` timesteps.

```
rolling_mean(t) = mean(features[max(0, t-w+1) .. t])
```

For `t >= w`: `rolling_mean = (cumsum[t] - cumsum[t-w]) / w`
For `t < w`: `rolling_mean = cumsum[t] / (t+1)` (cumulative mean)

Computed efficiently using cumulative sums, avoiding explicit loops over the window. Captures the recent level of a feature, which may differ substantially from its cumulative average later in the stay.

### 11. Rolling standard deviation

Standard deviation of feature values over the last `w` timesteps.

```
rolling_std(t) = sqrt(mean(x^2[window]) - rolling_mean(t)^2)
```

Uses the same `E[X^2] - E[X]^2` identity as the cumulative std, but restricted to the rolling window. Computed using cumulative sums of squared values.

Captures **recent** variability. A patient may have low cumulative std (stable overall) but high rolling std (acutely unstable in the last few hours), which is a more actionable signal for short-term prediction.

### 12. Rolling linear trend (slope)

Least-squares linear regression slope fitted to the feature values in the rolling window.

```
slope(t) = (sum(i * x[i]) - n * mean_i * mean_x) / (sum(i^2) - n * mean_i^2)
```

where `i = 0, 1, ..., w-1` indexes positions within the window, and `x[i]` are the corresponding feature values.

Computed via `np.einsum` for vectorised dot products across all samples and features simultaneously.

Captures the **direction and rate** of recent change. A positive slope means the feature is trending upward over the last `w` hours; a negative slope means it is declining. Unlike the diff feature (which only sees one-step changes and is sensitive to noise), the trend provides a smoothed estimate of the direction of change.

---

## Feature Count Summary

| Configuration | Features per timestep | Total (flattened) |
|---|---|---|
| Base only | `7 * n_raw + 1` | 619 |
| + Lag | `9 * n_raw + 1` | 825 |
| + Rolling | `10 * n_raw + 1` | 928 |
| + Lag + Rolling | `12 * n_raw + 1` | 1134 |

Where `n_raw` = number of original features (typically ~88 for END prediction with imaging). The `+1` accounts for the timestep index (a single scalar, not per-feature).

---

## Pipeline

```
Raw time series (n_patients, 72, n_raw)
    |
    v
aggregate_features_over_time()
    |-- raw values
    |-- cumulative mean
    |-- cumulative min
    |-- cumulative max
    |-- cumulative std
    |-- first-order differences
    |-- timestep index
    |-- [optional] lag-2, lag-3
    |-- [optional] rolling mean, rolling std, rolling trend
    |
    v
Concatenate along feature axis -> (n_patients, 72, n_total_features)
    |
    v
Flatten -> (n_patients * 72, n_total_features)
    |
    v
StandardScaler (in prepare_aggregate_dataset)
    |
    v
XGBoost input
```

"""
What each column of the aggregated XGB input means, and how to read it in clinical units.

Column order is the one produced by aggregate_features_over_time, n = number of raw features:

    [0:n) raw | mean | min | max | std | diff | hour | lag2 | lag3 | rolling mean | rolling std | rolling trend
                └──────── since admission ────────┘        └─ optional ─┘ └────────── optional, last 6h ───────┘

Columns are identified by position, never by name: raw features such as
'max_NIHSS' already start with what looks like an aggregation prefix.
"""
from enum import Enum
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd


class Block(Enum):
    RAW = 'current value'
    CUMULATIVE_MEAN = 'mean since admission'
    CUMULATIVE_MIN = 'minimum since admission'
    CUMULATIVE_MAX = 'maximum since admission'
    CUMULATIVE_STD = 'variability (SD) since admission'
    DIFF = 'change vs previous hour'
    HOUR = 'hours since admission'
    LAG2 = 'value 2h ago'
    LAG3 = 'value 3h ago'
    ROLLING_MEAN = 'mean over last 6h'
    ROLLING_STD = 'variability (SD) over last 6h'
    ROLLING_TREND = 'trend per hour over last 6h'


BASE_BLOCKS = [Block.RAW, Block.CUMULATIVE_MEAN, Block.CUMULATIVE_MIN, Block.CUMULATIVE_MAX,
               Block.CUMULATIVE_STD, Block.DIFF]
LAG_BLOCKS = [Block.LAG2, Block.LAG3]
ROLLING_BLOCKS = [Block.ROLLING_MEAN, Block.ROLLING_STD, Block.ROLLING_TREND]

# Differences and spreads of a z-scored variable only need rescaling; levels also need the mean added back
SCALE_ONLY_BLOCKS = {Block.CUMULATIVE_STD, Block.DIFF, Block.ROLLING_STD, Block.ROLLING_TREND}

N_HOURS = 72


class AggregatedFeature(NamedTuple):
    block: Block
    raw_feature: Optional[str]


def build_layout(raw_feature_names: List[str], n_columns: int) -> List[AggregatedFeature]:
    """
    Describe every aggregated column. Optional blocks are inferred from the column count.

    e.g. 103 raw features, 1134 columns = 11 x 103 + 1 -> base, lag and rolling blocks
    """
    n_raw = len(raw_feature_names)
    candidate_block_sets = [
        BASE_BLOCKS,
        BASE_BLOCKS + LAG_BLOCKS,
        BASE_BLOCKS + ROLLING_BLOCKS,
        BASE_BLOCKS + LAG_BLOCKS + ROLLING_BLOCKS,
    ]

    for blocks in candidate_block_sets:
        if len(blocks) * n_raw + 1 != n_columns:
            continue

        layout = []
        for block in blocks:
            layout += [AggregatedFeature(block, name) for name in raw_feature_names]

            # The hour column sits right after the base blocks
            if block == BASE_BLOCKS[-1]:
                layout.append(AggregatedFeature(Block.HOUR, None))

        return layout

    raise ValueError(f'{n_columns} columns do not match any known layout of {n_raw} raw features.')


class ClinicalUnits:
    """Converts z-scored values back to the units clinicians read (mmHg, mmol/L, NIHSS points...)."""

    def __init__(self, normalisation_parameters: pd.DataFrame):
        parameters = normalisation_parameters.set_index('variable')
        self._mean = parameters['original_mean'].to_dict()
        self._std = parameters['original_std'].to_dict()

    def is_normalised(self, raw_feature: str) -> bool:
        return raw_feature in self._std

    def convert(self, value: float, feature: AggregatedFeature) -> float:
        """e.g. systolic BP z-score 1.0 -> 140 + 23.7 = 163.7 mmHg; its hourly change 1.0 -> 23.7 mmHg"""
        if feature.block == Block.HOUR:
            return value * (N_HOURS - 1)

        # Binary variables were never normalised
        if not self.is_normalised(feature.raw_feature):
            return value

        rescaled = value * self._std[feature.raw_feature]
        if feature.block in SCALE_ONLY_BLOCKS:
            return rescaled

        return rescaled + self._mean[feature.raw_feature]

    def convert_raw(self, values: np.ndarray, raw_feature: str) -> np.ndarray:
        """Vectorised conversion of a raw (current value) series."""
        if not self.is_normalised(raw_feature):
            return values

        return values * self._std[raw_feature] + self._mean[raw_feature]

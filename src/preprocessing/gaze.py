"""
Gaze preprocessing.
The paper does not specify signal-level filtering for Gaze.
We simply pass through the loaded DataFrame; feature extraction
computes statistics directly on each 10-second window.
"""

import pandas as pd


def preprocess_gaze(sessions: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Forward sessions unchanged; feature extractor handles windowing."""
    return [s.copy() for s in sessions]

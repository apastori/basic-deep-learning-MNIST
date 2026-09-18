"""Provide array-shape validation helpers."""

import numpy as np


def assert_same_shape(x: np.ndarray, y: np.ndarray) -> None:
    """Raise an error if two arrays do not have the same shape."""
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: expected {x.shape}, got {y.shape}")

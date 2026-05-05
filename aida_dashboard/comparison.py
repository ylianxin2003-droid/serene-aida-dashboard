from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComparisonResult:
    difference: np.ndarray
    mean_difference: float
    mean_absolute_difference: float
    rmse: float
    min_difference: float
    max_difference: float
    baseline_mean: float
    candidate_mean: float


def compare_grids(baseline: np.ndarray, candidate: np.ndarray) -> ComparisonResult:
    if baseline.shape != candidate.shape:
        raise ValueError(f"Cannot compare grids with different shapes: {baseline.shape} and {candidate.shape}")

    difference = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    return ComparisonResult(
        difference=difference,
        mean_difference=float(np.nanmean(difference)),
        mean_absolute_difference=float(np.nanmean(np.abs(difference))),
        rmse=float(np.sqrt(np.nanmean(difference**2))),
        min_difference=float(np.nanmin(difference)),
        max_difference=float(np.nanmax(difference)),
        baseline_mean=float(np.nanmean(baseline)),
        candidate_mean=float(np.nanmean(candidate)),
    )

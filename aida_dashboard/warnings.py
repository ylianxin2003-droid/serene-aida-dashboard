from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DashboardWarning:
    kind: str
    severity: str
    message: str
    count: int
    peak_value: float
    peak_latitude: float
    peak_longitude: float
    units: str


def evaluate_warnings(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    variables: dict[str, np.ndarray],
    tec_high: float,
    muf_low: float,
    fof2_low: float,
) -> list[DashboardWarning]:
    warnings: list[DashboardWarning] = []

    if "TEC" in variables:
        warning = _high_threshold_warning(
            "GNSS positioning risk",
            "High TEC may increase GNSS delay and positioning uncertainty.",
            latitudes,
            longitudes,
            variables["TEC"],
            tec_high,
            "TECU",
        )
        if warning:
            warnings.append(warning)

    if "MUF3000F2" in variables:
        warning = _low_threshold_warning(
            "HF communication risk",
            "Low MUF3000F2 indicates possible degradation of long-range HF links.",
            latitudes,
            longitudes,
            variables["MUF3000F2"],
            muf_low,
            "MHz",
        )
        if warning:
            warnings.append(warning)

    if "foF2" in variables:
        warning = _low_threshold_warning(
            "HF propagation degradation",
            "Low foF2 indicates reduced F2-layer critical frequency support.",
            latitudes,
            longitudes,
            variables["foF2"],
            fof2_low,
            "MHz",
        )
        if warning:
            warnings.append(warning)

    return warnings


def _high_threshold_warning(
    kind: str,
    message: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    threshold: float,
    units: str,
) -> DashboardWarning | None:
    mask = values >= threshold
    if not np.any(mask):
        return None

    masked_values = np.where(mask, values, np.nan)
    flat_index = int(np.nanargmax(masked_values))
    row, col = np.unravel_index(flat_index, values.shape)
    peak = float(values[row, col])
    return DashboardWarning(
        kind=kind,
        severity=_severity(float(np.nanmax(values)) / threshold),
        message=message,
        count=int(np.count_nonzero(mask)),
        peak_value=peak,
        peak_latitude=float(latitudes[row]),
        peak_longitude=float(longitudes[col]),
        units=units,
    )


def _low_threshold_warning(
    kind: str,
    message: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    threshold: float,
    units: str,
) -> DashboardWarning | None:
    mask = values <= threshold
    if not np.any(mask):
        return None

    masked_values = np.where(mask, values, np.nan)
    flat_index = int(np.nanargmin(masked_values))
    row, col = np.unravel_index(flat_index, values.shape)
    peak = float(values[row, col])
    ratio = threshold / max(peak, 1e-9)
    return DashboardWarning(
        kind=kind,
        severity=_severity(ratio),
        message=message,
        count=int(np.count_nonzero(mask)),
        peak_value=peak,
        peak_latitude=float(latitudes[row]),
        peak_longitude=float(longitudes[col]),
        units=units,
    )


def _severity(ratio: float) -> str:
    if ratio >= 1.5:
        return "Severe"
    if ratio >= 1.2:
        return "Moderate"
    return "Watch"

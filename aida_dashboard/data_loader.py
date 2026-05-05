from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import h5py
import numpy as np


FORECAST_LABELS = {
    "030": "+0.5 h forecast",
    "090": "+1.5 h forecast",
    "180": "+3.0 h forecast",
    "360": "+6.0 h forecast",
}


@dataclass(frozen=True)
class ProductPair:
    label: str
    sort_key: int
    latency: str
    forecast_minutes: int
    param_file: Path | None = None
    ne_file: Path | None = None


@dataclass(frozen=True)
class Param2DProduct:
    path: Path
    label: str
    time_utc: str
    version: str
    latitudes: np.ndarray
    longitudes: np.ndarray
    variables: dict[str, np.ndarray]


@dataclass(frozen=True)
class Ne3DProduct:
    path: Path
    label: str
    time_utc: str
    version: str
    latitudes: np.ndarray
    longitudes: np.ndarray
    altitudes: np.ndarray
    ne: np.ndarray
    tec: np.ndarray

    def ne_at_altitude(self, altitude_km: float) -> np.ndarray:
        index = int(np.nanargmin(np.abs(self.altitudes - altitude_km)))
        return self.ne[:, :, index]


def build_forecast_label(filename: str) -> str:
    latency = product_latency(filename)
    match = re.search(r"_f(\d{3})_", filename)
    if not match:
        return f"{latency} Analysis"
    lead = FORECAST_LABELS.get(match.group(1), f"+{int(match.group(1)) / 60:.1f} h forecast")
    return f"{latency} {lead}"


def product_latency(filename: str) -> str:
    if re.search(r"_(r)(?:_|_f)", filename):
        return "Rapid"
    if re.search(r"_(u)(?:_|_f)", filename):
        return "Ultra-Rapid"
    return "Unknown"


def forecast_sort_key(filename: str) -> int:
    match = re.search(r"_f(\d{3})_", filename)
    if not match:
        return 0
    return int(match.group(1))


def discover_products(data_dir: str | Path) -> list[ProductPair]:
    data_path = Path(data_dir)
    pairs: dict[tuple[str, int], ProductPair] = {}

    for path in data_path.glob("*.h5"):
        sort_key = forecast_sort_key(path.name)
        latency = product_latency(path.name)
        key = (latency, sort_key)
        current = pairs.get(
            key,
            ProductPair(
                build_forecast_label(path.name),
                _latency_sort_key(latency) * 1000 + sort_key,
                latency,
                sort_key,
            ),
        )
        if path.name.startswith("param_2d"):
            current = ProductPair(
                current.label,
                current.sort_key,
                current.latency,
                current.forecast_minutes,
                param_file=path,
                ne_file=current.ne_file,
            )
        elif path.name.startswith("Ne_3d"):
            current = ProductPair(
                current.label,
                current.sort_key,
                current.latency,
                current.forecast_minutes,
                param_file=current.param_file,
                ne_file=path,
            )
        pairs[key] = current

    return sorted(pairs.values(), key=lambda product: product.sort_key)


def load_param2d_product(path: str | Path) -> Param2DProduct:
    file_path = Path(path)
    with h5py.File(file_path, "r") as open_file:
        latitudes = open_file["Latitudes"][()]
        longitudes = open_file["Longitudes"][()]
        variables = {
            key: _lon_lat_to_lat_lon(open_file[key][()])
            for key in ["TEC", "MUF3000F2", "foF2", "NmF2", "hmF2"]
            if key in open_file
        }

    return Param2DProduct(
        path=file_path,
        label=build_forecast_label(file_path.name),
        time_utc=_time_from_filename(file_path.name),
        version=f"AIDA {product_latency(file_path.name)} parameter product",
        latitudes=latitudes,
        longitudes=longitudes,
        variables=variables,
    )


def load_ne3d_product(path: str | Path) -> Ne3DProduct:
    file_path = Path(path)
    with h5py.File(file_path, "r") as open_file:
        latitudes = open_file["Latitudes"][()]
        longitudes = open_file["Longitudes"][()]
        altitudes = open_file["Altitudes"][()]
        ne = np.transpose(open_file["Ne"][()], (1, 0, 2))
        tec = _lon_lat_to_lat_lon(open_file["TEC"][()])
        time_utc = _decode_scalar(open_file["t0"][()])
        version = _decode_scalar(open_file["version"][()])

    return Ne3DProduct(
        path=file_path,
        label=build_forecast_label(file_path.name),
        time_utc=_format_product_time(time_utc),
        version=version,
        latitudes=latitudes,
        longitudes=longitudes,
        altitudes=altitudes,
        ne=ne,
        tec=tec,
    )


def _lon_lat_to_lat_lon(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).T


def _decode_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _format_product_time(value: str) -> str:
    parsed = datetime.strptime(value, "%d/%m/%Y, %H:%M:%S")
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")


def _time_from_filename(filename: str) -> str:
    match = re.search(r"_(\d{6})_(\d{6})\.h5$", filename)
    if not match:
        return "Unknown UTC"
    parsed = datetime.strptime("20" + match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")


def _latency_sort_key(latency: str) -> int:
    if latency == "Rapid":
        return 0
    if latency == "Ultra-Rapid":
        return 1
    return 2

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class IonexTecMap:
    path: Path
    model: str
    time_utc: str
    latitudes: np.ndarray
    longitudes: np.ndarray
    tec: np.ndarray


def discover_ionex_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.INX"))


def load_ionex_tec(path: str | Path) -> IonexTecMap:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    exponent = _header_exponent(lines)
    model = _header_model(lines)
    time_utc = _header_time(lines)

    rows: list[tuple[float, list[float]]] = []
    index = _first_tec_map_index(lines)
    while index < len(lines):
        line = lines[index]
        if "END OF TEC MAP" in line:
            break
        if "LAT/LON1/LON2/DLON/H" in line:
            lat, lon1, lon2, dlon = _parse_lat_lon_header(line)
            expected = int(round((lon2 - lon1) / dlon)) + 1
            values: list[float] = []
            index += 1
            while index < len(lines) and len(values) < expected:
                if "LAT/LON1/LON2/DLON/H" in lines[index] or "END OF TEC MAP" in lines[index]:
                    break
                values.extend(float(token) * (10**exponent) for token in lines[index].split())
                index += 1
            rows.append((lat, values[:expected]))
            continue
        index += 1

    if not rows:
        raise ValueError(f"No TEC map rows found in {file_path}")

    latitudes = np.array([lat for lat, _ in rows])
    width = min(len(values) for _, values in rows)
    longitudes = np.arange(-180.0, -180.0 + 2.5 * width, 2.5)
    tec = np.array([values[:width] for _, values in rows], dtype=float)
    return IonexTecMap(file_path, model, time_utc, latitudes, longitudes, tec)


def _header_exponent(lines: list[str]) -> int:
    for line in lines:
        if "EXPONENT" in line:
            return int(line[:6])
    return 0


def _header_model(lines: list[str]) -> str:
    for line in lines[:20]:
        if "PGM / RUN BY / DATE" in line:
            model = line[:20].strip()
            return model.replace("AIDA R ", "AIDA Rapid ").replace("AIDA U ", "AIDA Ultra-Rapid ")
    return "IONEX TEC"


def _header_time(lines: list[str]) -> str:
    for line in lines:
        if "EPOCH OF FIRST MAP" in line:
            values = [int(value) for value in line[:43].split()[:6]]
            parsed = datetime(*values)
            return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
    return "Unknown UTC"


def _first_tec_map_index(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if "START OF TEC MAP" in line:
            return index + 1
    raise ValueError("No START OF TEC MAP marker found")


def _parse_lat_lon_header(line: str) -> tuple[float, float, float, float]:
    values = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", line[:60])]
    if len(values) < 4:
        raise ValueError(f"Cannot parse IONEX latitude/longitude header: {line}")
    return values[0], values[1], values[2], values[3]

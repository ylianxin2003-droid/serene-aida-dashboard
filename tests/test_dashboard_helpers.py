from pathlib import Path

import numpy as np

from aida_dashboard.data_loader import (
    build_forecast_label,
    discover_products,
    load_ne3d_product,
    load_param2d_product,
)
from aida_dashboard.comparison import compare_grids
from aida_dashboard.ionex import load_ionex_tec
from aida_dashboard.warnings import evaluate_warnings


DATA_DIR = Path(__file__).resolve().parents[1] / "local_data"


def test_discover_products_pairs_param_and_ne_files():
    products = discover_products(DATA_DIR)

    labels = [product.label for product in products]
    assert labels == [
        "Rapid Analysis",
        "Rapid +0.5 h forecast",
        "Rapid +1.5 h forecast",
        "Rapid +3.0 h forecast",
        "Rapid +6.0 h forecast",
        "Ultra-Rapid Analysis",
        "Ultra-Rapid +0.5 h forecast",
        "Ultra-Rapid +1.5 h forecast",
        "Ultra-Rapid +3.0 h forecast",
        "Ultra-Rapid +6.0 h forecast",
    ]
    assert all(product.param_file and product.ne_file for product in products)
    assert {product.latency for product in products} == {"Rapid", "Ultra-Rapid"}


def test_load_param2d_product_exposes_expected_fields_and_grid():
    product = load_param2d_product(DATA_DIR / "param_2d_u_260501_132500.h5")

    assert product.time_utc == "2026-05-01 13:25:00 UTC"
    assert product.version == "AIDA Ultra-Rapid parameter product"
    assert product.latitudes.shape == (181,)
    assert product.longitudes.shape == (360,)
    assert set(["TEC", "MUF3000F2", "foF2", "NmF2", "hmF2"]).issubset(product.variables)
    assert product.variables["TEC"].shape == (181, 360)
    assert np.nanmax(product.variables["TEC"]) > 70


def test_load_ne3d_product_exposes_altitude_slices():
    product = load_ne3d_product(DATA_DIR / "Ne_3d_u_260501_132500.h5")

    assert product.time_utc == "2026-05-01 13:25:00 UTC"
    assert product.version == "AIDA Ultra Rapid v1.1.1"
    assert product.altitudes[0] == 50
    assert product.ne.shape == (181, 360, 164)
    slice_300 = product.ne_at_altitude(300)
    assert slice_300.shape == (181, 360)
    assert np.nanmax(slice_300) > 1e11


def test_build_forecast_label_decodes_filename_codes():
    assert build_forecast_label("param_2d_u_260501_132500.h5") == "Ultra-Rapid Analysis"
    assert build_forecast_label("param_2d_r_260501_115500.h5") == "Rapid Analysis"
    assert build_forecast_label("param_2d_u_f030_260501_135000.h5") == "Ultra-Rapid +0.5 h forecast"
    assert build_forecast_label("param_2d_r_f090_260501_132000.h5") == "Rapid +1.5 h forecast"


def test_compare_grids_reports_bias_rmse_and_extremes():
    baseline = np.array([[1.0, 2.0], [3.0, 4.0]])
    candidate = np.array([[2.0, 2.0], [1.0, 8.0]])

    result = compare_grids(baseline, candidate)

    assert result.mean_difference == 0.75
    assert np.isclose(result.rmse, np.sqrt(21 / 4))
    assert result.max_difference == 4.0
    assert result.min_difference == -2.0


def test_load_ionex_tec_reads_serene_tec_grid():
    ionex = load_ionex_tec(DATA_DIR / "BHA0OPSNRT_20261211155_05M_05M_GIM.INX")

    assert ionex.model == "AIDA Rapid v1.1.1"
    assert ionex.time_utc == "2026-05-01 11:55:00 UTC"
    assert ionex.latitudes.shape == (73,)
    assert ionex.longitudes.shape == (144,)
    assert ionex.tec.shape == (73, 144)
    assert np.nanmax(ionex.tec) > 70


def test_evaluate_warnings_returns_highest_risk_regions():
    latitudes = np.array([-1.0, 0.0, 1.0])
    longitudes = np.array([10.0, 11.0])
    variables = {
        "TEC": np.array([[10.0, 91.0], [20.0, 95.0], [30.0, 40.0]]),
        "MUF3000F2": np.array([[8.0, 4.5], [6.0, 7.0], [5.5, 10.0]]),
        "foF2": np.array([[4.0, 2.5], [3.5, 4.5], [5.0, 6.0]]),
    }

    warnings = evaluate_warnings(
        latitudes,
        longitudes,
        variables,
        tec_high=80.0,
        muf_low=5.0,
        fof2_low=3.0,
    )

    assert [warning.kind for warning in warnings] == [
        "GNSS positioning risk",
        "HF communication risk",
        "HF propagation degradation",
    ]
    assert warnings[0].count == 2
    assert warnings[0].peak_value == 95.0
    assert warnings[0].peak_latitude == 0.0
    assert warnings[0].peak_longitude == 11.0

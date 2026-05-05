from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from aida import AIDAState
from aida.api import default_api_config, downloadOutput
from aida_dashboard.comparison import compare_grids
from aida_dashboard.data_loader import (
    Param2DProduct,
    ProductPair,
    discover_products,
    load_ne3d_product,
    load_param2d_product,
)
from aida_dashboard.ionex import discover_ionex_files, load_ionex_tec
from aida_dashboard.warnings import evaluate_warnings


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "local_data"
DEFAULT_API_CACHE_DIR = ROOT / "api_cache"

PARAMETER_UNITS = {
    "TEC": "TECU",
    "MUF3000F2": "MHz",
    "foF2": "MHz",
    "NmF2": "m^-3",
    "hmF2": "km",
    "Ne": "m^-3",
}

PARAMETER_LABELS = {
    "TEC": "Total Electron Content",
    "MUF3000F2": "MUF3000F2",
    "foF2": "F2 critical frequency",
    "NmF2": "F2 peak electron density",
    "hmF2": "F2 peak height",
    "Ne": "Electron density at selected altitude",
}


def main() -> None:
    st.set_page_config(page_title="AIDA Space Weather Dashboard", layout="wide")
    st.title("AIDA Space Weather Dashboard")
    st.caption("Local prototype for ionospheric monitoring, forecast comparison, and aviation risk analysis")

    data_source = _sidebar_data_source()
    thresholds = _sidebar_thresholds()
    tabs = st.tabs(["Overview", "Forecast Comparison", "Product Comparison", "IONEX TEC", "Risk Analysis"])

    if data_source.startswith("SERENE"):
        latency = "ultra" if "Ultra-Rapid" in data_source else "rapid"
        if st.sidebar.button("Refresh SERENE latest"):
            _latest_serene_param_product.clear()
        try:
            product, param_product = _latest_serene_param_product(latency)
        except Exception as error:
            st.error(f"Could not load SERENE latest data: {error}")
            st.stop()

        with tabs[0]:
            _api_overview_tab(product, param_product, thresholds)
        with tabs[1]:
            st.info("Forecast comparison currently uses the local sample product set.")
        with tabs[2]:
            st.info("Product comparison currently uses the local sample product set.")
        with tabs[3]:
            st.info("IONEX comparison currently uses local IONEX files.")
        with tabs[4]:
            _api_risk_analysis_tab(product, param_product, thresholds)
        return

    data_dir = _sidebar_data_dir()
    products = discover_products(data_dir)
    ionex_files = discover_ionex_files(data_dir)
    if not products:
        st.error(f"No AIDA .h5 products found in {data_dir}")
        st.stop()

    with tabs[0]:
        _overview_tab(products, thresholds)
    with tabs[1]:
        _forecast_comparison_tab(products)
    with tabs[2]:
        _product_comparison_tab(products)
    with tabs[3]:
        _ionex_tab(products, ionex_files)
    with tabs[4]:
        _risk_analysis_tab(products, thresholds)


def _overview_tab(products: list[ProductPair], thresholds: tuple[float, float, float]) -> None:
    selected_pair = _product_selector("Overview product", products)
    param_product = load_param2d_product(selected_pair.param_file) if selected_pair.param_file else None
    ne_product = load_ne3d_product(selected_pair.ne_file) if selected_pair.ne_file else None

    available_parameters = []
    if param_product:
        available_parameters.extend(param_product.variables.keys())
    if ne_product:
        available_parameters.append("Ne")

    selected_parameter = st.selectbox(
        "Displayed parameter",
        available_parameters,
        format_func=lambda key: f"{key} - {PARAMETER_LABELS.get(key, key)}",
    )

    altitude = None
    if selected_parameter == "Ne" and ne_product:
        altitude = st.slider(
            "Ne altitude layer (km)",
            min_value=float(ne_product.altitudes.min()),
            max_value=float(ne_product.altitudes.max()),
            value=300.0,
            step=10.0,
        )

    latitudes, longitudes, values, time_utc, version = _selected_grid(
        selected_parameter,
        param_product,
        ne_product,
        altitude,
    )
    _render_header(selected_pair, time_utc, version)

    left, right = st.columns([2.3, 1.0], gap="large")
    with left:
        _render_map(selected_parameter, latitudes, longitudes, values, altitude)
    with right:
        _render_statistics(selected_parameter, values)
        if param_product:
            _render_warnings(_warnings_for_product(param_product, thresholds))


def _forecast_comparison_tab(products: list[ProductPair]) -> None:
    st.subheader("Forecast lead-time comparison")
    latency_options = sorted({product.latency for product in products})
    latency = st.selectbox("Latency stream", latency_options, index=latency_options.index("Ultra-Rapid") if "Ultra-Rapid" in latency_options else 0)
    parameter = st.selectbox("Parameter", ["TEC", "MUF3000F2", "foF2", "NmF2", "hmF2"], key="forecast_parameter")

    stream = [product for product in products if product.latency == latency and product.param_file]
    analysis = next((product for product in stream if product.forecast_minutes == 0), None)
    forecasts = [product for product in stream if product.forecast_minutes > 0]
    if not analysis or not forecasts:
        st.info("Need one analysis file and at least one forecast file for this comparison.")
        return

    baseline = load_param2d_product(analysis.param_file)
    rows = []
    comparisons = {}
    for forecast in forecasts:
        candidate = load_param2d_product(forecast.param_file)
        result = compare_grids(baseline.variables[parameter], candidate.variables[parameter])
        comparisons[forecast.label] = (candidate, result)
        rows.append(_comparison_row(forecast.label, candidate.time_utc, result, parameter))

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    selected_label = st.selectbox("Difference map", list(comparisons), key="forecast_diff")
    candidate, result = comparisons[selected_label]
    _comparison_metrics(result, parameter)
    _render_difference_map(
        f"{selected_label} minus {analysis.label}: {parameter}",
        baseline.latitudes,
        baseline.longitudes,
        result.difference,
        parameter,
    )


def _product_comparison_tab(products: list[ProductPair]) -> None:
    st.subheader("Product-to-product comparison")
    parameter = st.selectbox("Parameter", ["TEC", "MUF3000F2", "foF2", "NmF2", "hmF2"], key="product_parameter")
    baseline_pair = _product_selector("Baseline product", products, key="baseline_product")
    candidate_pair = _product_selector("Candidate product", products, key="candidate_product")

    if not baseline_pair.param_file or not candidate_pair.param_file:
        st.info("Both selected products need a 2D parameter file.")
        return

    baseline = load_param2d_product(baseline_pair.param_file)
    candidate = load_param2d_product(candidate_pair.param_file)
    result = compare_grids(baseline.variables[parameter], candidate.variables[parameter])

    _comparison_metrics(result, parameter)
    _render_difference_map(
        f"{candidate_pair.label} minus {baseline_pair.label}: {parameter}",
        baseline.latitudes,
        baseline.longitudes,
        result.difference,
        parameter,
    )
    st.markdown(_comparison_text(candidate_pair.label, baseline_pair.label, result, parameter))


def _ionex_tab(products: list[ProductPair], ionex_files: list[Path]) -> None:
    st.subheader("IONEX TEC comparison")
    if not ionex_files:
        st.info("No .INX IONEX files found in local_data.")
        return

    ionex_path = st.selectbox("IONEX file", ionex_files, format_func=lambda path: path.name)
    ionex = load_ionex_tec(ionex_path)
    st.metric("IONEX model", ionex.model)
    st.metric("IONEX time", ionex.time_utc.replace(" UTC", ""))

    left, right = st.columns([1.5, 1.0], gap="large")
    with left:
        _render_map("TEC", ionex.latitudes, ionex.longitudes, ionex.tec, altitude=None, title=f"IONEX TEC: {ionex_path.name}")
    with right:
        _render_statistics("TEC", ionex.tec)

    candidate_pair = _product_selector(
        "AIDA product sampled onto IONEX grid",
        [product for product in products if product.param_file],
        key="ionex_candidate",
    )
    candidate = load_param2d_product(candidate_pair.param_file)
    sampled = _sample_to_grid(
        candidate.latitudes,
        candidate.longitudes,
        candidate.variables["TEC"],
        ionex.latitudes,
        ionex.longitudes,
    )
    result = compare_grids(ionex.tec, sampled)
    _comparison_metrics(result, "TEC")
    _render_difference_map(
        f"{candidate_pair.label} sampled TEC minus IONEX TEC",
        ionex.latitudes,
        ionex.longitudes,
        result.difference,
        "TEC",
    )


def _risk_analysis_tab(products: list[ProductPair], thresholds: tuple[float, float, float]) -> None:
    st.subheader("Aviation-oriented risk analysis")
    rows = []
    for product in products:
        if not product.param_file:
            continue
        param = load_param2d_product(product.param_file)
        warnings = _warnings_for_product(param, thresholds)
        rows.append(
            {
                "Product": product.label,
                "Time (UTC)": param.time_utc.replace(" UTC", ""),
                "Warnings": len(warnings),
                "TEC max": float(np.nanmax(param.variables["TEC"])),
                "MUF min": float(np.nanmin(param.variables["MUF3000F2"])),
                "foF2 min": float(np.nanmin(param.variables["foF2"])),
                "Summary": _warning_summary(warnings),
            }
        )

    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.markdown(_overall_analysis_text(table))


def _api_overview_tab(product: ProductPair, param_product: Param2DProduct, thresholds: tuple[float, float, float]) -> None:
    available_parameters = list(param_product.variables.keys())
    selected_parameter = st.selectbox(
        "Displayed parameter",
        available_parameters,
        format_func=lambda key: f"{key} - {PARAMETER_LABELS.get(key, key)}",
    )

    values = param_product.variables[selected_parameter]
    _render_header(product, param_product.time_utc, param_product.version)

    left, right = st.columns([2.3, 1.0], gap="large")
    with left:
        _render_map(selected_parameter, param_product.latitudes, param_product.longitudes, values, altitude=None)
    with right:
        _render_statistics(selected_parameter, values)
        _render_warnings(_warnings_for_product(param_product, thresholds))


def _api_risk_analysis_tab(product: ProductPair, param_product: Param2DProduct, thresholds: tuple[float, float, float]) -> None:
    st.subheader("Aviation-oriented risk analysis")
    warnings = _warnings_for_product(param_product, thresholds)
    table = pd.DataFrame(
        [
            {
                "Product": product.label,
                "Time (UTC)": param_product.time_utc.replace(" UTC", ""),
                "Warnings": len(warnings),
                "TEC max": float(np.nanmax(param_product.variables["TEC"])),
                "MUF min": float(np.nanmin(param_product.variables["MUF3000F2"])),
                "foF2 min": float(np.nanmin(param_product.variables["foF2"])),
                "Summary": _warning_summary(warnings),
            }
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.markdown(_overall_analysis_text(table))


def _sidebar_data_source() -> str:
    st.sidebar.subheader("Data source")
    return st.sidebar.radio(
        "AIDA source",
        _data_source_options(DEFAULT_DATA_DIR),
    )


def _data_source_options(local_data_dir: str | Path) -> list[str]:
    options = ["SERENE Latest Ultra-Rapid", "SERENE Latest Rapid"]
    if Path(local_data_dir).exists() and any(Path(local_data_dir).glob("*.h5")):
        options.append("Local demo data")
    return options


def _sidebar_data_dir() -> Path:
    data_dir_text = st.sidebar.text_input("Local AIDA data folder", str(DEFAULT_DATA_DIR))
    return Path(data_dir_text).expanduser()


def _sidebar_thresholds() -> tuple[float, float, float]:
    st.sidebar.subheader("Warning thresholds")
    tec_high = st.sidebar.slider("High TEC threshold (TECU)", 20.0, 150.0, 80.0, 5.0)
    muf_low = st.sidebar.slider("Low MUF3000F2 threshold (MHz)", 1.0, 20.0, 5.0, 0.5)
    fof2_low = st.sidebar.slider("Low foF2 threshold (MHz)", 1.0, 15.0, 3.0, 0.5)
    return tec_high, muf_low, fof2_low


def _product_selector(label: str, products: list[ProductPair], key: str | None = None) -> ProductPair:
    label_to_product = {product.label: product for product in products}
    selected_label = st.selectbox(label, list(label_to_product), key=key)
    return label_to_product[selected_label]


def _selected_grid(selected_parameter, param_product, ne_product, altitude):
    if selected_parameter == "Ne":
        values = ne_product.ne_at_altitude(altitude)
        return ne_product.latitudes, ne_product.longitudes, values, ne_product.time_utc, ne_product.version
    values = param_product.variables[selected_parameter]
    return param_product.latitudes, param_product.longitudes, values, param_product.time_utc, param_product.version


@st.cache_data(ttl=300, show_spinner="Fetching latest SERENE AIDA data...")
def _latest_serene_param_product(latency: str) -> tuple[ProductPair, Param2DProduct]:
    output_file = downloadOutput(_serene_api_config_or_raise(), time="latest", model="AIDA", latency=latency)
    return _param_product_from_aida_output(output_file, latency)


def _serene_api_config_or_raise() -> dict:
    config = _serene_api_config()
    if config is not None:
        return config
    if default_api_config().exists():
        return None
    raise RuntimeError(_missing_serene_config_message())


def _serene_api_config() -> dict | None:
    token = _secret_value("SERENE_API_TOKEN") or _secret_value("serene.api_token") or os.environ.get("SERENE_API_TOKEN")
    if not token:
        return None
    return _serene_api_config_from_token(token, DEFAULT_API_CACHE_DIR)


def _missing_serene_config_message() -> str:
    return (
        "SERENE API token is not configured. For Streamlit Cloud, add "
        'SERENE_API_TOKEN = "your-token" in the app Secrets. For local runs, '
        "set the SERENE_API_TOKEN environment variable or configure "
        "~/.config/aida/api_config.ini."
    )


def _secret_value(name: str) -> str | None:
    try:
        value = st.secrets
        for part in name.split("."):
            value = value[part]
        return str(value)
    except Exception:
        return None


def _serene_api_config_from_token(token: str, cache_folder: str | Path = DEFAULT_API_CACHE_DIR) -> dict:
    cache_path = Path(cache_folder)
    cache_path.mkdir(parents=True, exist_ok=True)
    return {
        "api": {"token": token, "timeout": "30"},
        "cache": {"folder": str(cache_path), "subfolder": "{yyyy}/{doy}/"},
    }


def _param_product_from_aida_output(path: str | Path, latency: str) -> tuple[ProductPair, Param2DProduct]:
    file_path = Path(path)
    model = AIDAState()
    model.readFile(file_path)

    latitudes = np.linspace(-90.0, 90.0, 91)
    longitudes = np.linspace(0.0, 358.0, 180)
    output = model.calc(
        lat=latitudes,
        lon=longitudes,
        alt=None,
        grid="3D",
        collapse_particles=True,
        TEC=True,
        MUF3000=True,
    )
    variables = {
        "TEC": np.asarray(output["TEC"].values).T,
        "MUF3000F2": np.asarray(output["MUF3000"].values).T,
        "foF2": np.asarray(output["foF2"].values).T,
        "NmF2": np.asarray(output["NmF2"].values).T,
        "hmF2": np.asarray(output["hmF2"].values).T,
    }

    latency_label = "Ultra-Rapid" if latency == "ultra" else "Rapid"
    label = f"SERENE Latest {latency_label}"
    time_utc = datetime.fromtimestamp(float(model.Time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    product = ProductPair(
        label=label,
        sort_key=0,
        latency=latency_label,
        forecast_minutes=0,
        param_file=file_path,
    )
    param_product = Param2DProduct(
        path=file_path,
        label=label,
        time_utc=time_utc,
        version=str(model.Version),
        latitudes=latitudes,
        longitudes=longitudes,
        variables=variables,
    )
    return product, param_product


def _warnings_for_product(param_product, thresholds: tuple[float, float, float]):
    tec_high, muf_low, fof2_low = thresholds
    return evaluate_warnings(
        param_product.latitudes,
        param_product.longitudes,
        param_product.variables,
        tec_high=tec_high,
        muf_low=muf_low,
        fof2_low=fof2_low,
    )


def _render_header(product: ProductPair, time_utc: str, version: str) -> None:
    cols = st.columns(4)
    cols[0].metric("Product", product.label)
    cols[1].metric("Time", time_utc.replace(" UTC", ""))
    cols[2].metric("Model", version)
    cols[3].metric("Files", _file_status(product))


def _file_status(product: ProductPair) -> str:
    if product.param_file and product.param_file.name.startswith("output_"):
        return "SERENE API"
    if product.param_file and product.ne_file:
        return "2D + 3D"
    if product.param_file:
        return "2D only"
    if product.ne_file:
        return "3D only"
    return "Missing"


def _render_map(parameter: str, latitudes: np.ndarray, longitudes: np.ndarray, values: np.ndarray, altitude, title: str | None = None) -> None:
    title = title or PARAMETER_LABELS.get(parameter, parameter)
    if parameter == "Ne":
        title = f"{title} ({altitude:.0f} km)"

    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    mesh = ax.pcolormesh(longitudes, latitudes, values, shading="auto", cmap=_cmap(parameter))
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_xlim(float(np.nanmin(longitudes)), float(np.nanmax(longitudes)))
    ax.set_ylim(float(np.nanmin(latitudes)), float(np.nanmax(latitudes)))
    ax.grid(color="white", alpha=0.25, linewidth=0.6)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(PARAMETER_UNITS.get(parameter, ""))
    st.pyplot(fig, clear_figure=True)


def _render_difference_map(title: str, latitudes: np.ndarray, longitudes: np.ndarray, values: np.ndarray, parameter: str) -> None:
    limit = float(np.nanmax(np.abs(values)))
    if limit == 0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    mesh = ax.pcolormesh(longitudes, latitudes, values, shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(color="white", alpha=0.25, linewidth=0.6)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f"Difference ({PARAMETER_UNITS.get(parameter, '')})")
    st.pyplot(fig, clear_figure=True)


def _cmap(parameter: str) -> str:
    if parameter in {"MUF3000F2", "foF2"}:
        return "viridis"
    if parameter in {"NmF2", "Ne"}:
        return "magma"
    if parameter == "hmF2":
        return "cividis"
    return "plasma"


def _render_statistics(parameter: str, values: np.ndarray) -> None:
    units = PARAMETER_UNITS.get(parameter, "")
    st.subheader("Field statistics")
    st.metric("Minimum", _format_value(float(np.nanmin(values)), units))
    st.metric("Mean", _format_value(float(np.nanmean(values)), units))
    st.metric("Maximum", _format_value(float(np.nanmax(values)), units))


def _comparison_metrics(result, parameter: str) -> None:
    units = PARAMETER_UNITS.get(parameter, "")
    cols = st.columns(4)
    cols[0].metric("Bias", _format_value(result.mean_difference, units))
    cols[1].metric("MAE", _format_value(result.mean_absolute_difference, units))
    cols[2].metric("RMSE", _format_value(result.rmse, units))
    cols[3].metric("Range", f"{_format_value(result.min_difference, units)} to {_format_value(result.max_difference, units)}")


def _comparison_row(label: str, time_utc: str, result, parameter: str) -> dict[str, object]:
    return {
        "Forecast": label,
        "Time (UTC)": time_utc.replace(" UTC", ""),
        "Bias": result.mean_difference,
        "MAE": result.mean_absolute_difference,
        "RMSE": result.rmse,
        "Min diff": result.min_difference,
        "Max diff": result.max_difference,
        "Units": PARAMETER_UNITS.get(parameter, ""),
    }


def _render_warnings(warnings) -> None:
    st.subheader("ICAO-style warnings")
    if not warnings:
        st.success("No threshold exceedances for the selected warning settings.")
        return

    for warning in warnings:
        box = st.error if warning.severity == "Severe" else st.warning if warning.severity == "Moderate" else st.info
        box(
            f"{warning.severity}: {warning.kind}\n\n"
            f"{warning.message}\n\n"
            f"Peak/lowest value: {warning.peak_value:.2f} {warning.units} "
            f"near {warning.peak_latitude:.1f} deg, {warning.peak_longitude:.1f} deg. "
            f"Flagged grid cells: {warning.count}."
        )


def _format_value(value: float, units: str) -> str:
    if abs(value) >= 1e5:
        rendered = f"{value:.3e}"
    else:
        rendered = f"{value:.2f}"
    return f"{rendered} {units}".strip()


def _sample_to_grid(src_latitudes, src_longitudes, src_values, target_latitudes, target_longitudes) -> np.ndarray:
    lat_indices = np.searchsorted(src_latitudes, target_latitudes)
    lon_indices = np.searchsorted(src_longitudes, target_longitudes)
    lat_indices = np.clip(lat_indices, 0, len(src_latitudes) - 1)
    lon_indices = np.clip(lon_indices, 0, len(src_longitudes) - 1)
    return src_values[np.ix_(lat_indices, lon_indices)]


def _warning_summary(warnings) -> str:
    if not warnings:
        return "No threshold exceedance"
    return "; ".join(f"{warning.severity} {warning.kind}" for warning in warnings)


def _comparison_text(candidate_label: str, baseline_label: str, result, parameter: str) -> str:
    direction = "higher" if result.mean_difference > 0 else "lower"
    units = PARAMETER_UNITS.get(parameter, "")
    return (
        f"**Interpretation:** on average, `{candidate_label}` is "
        f"`{abs(result.mean_difference):.2f} {units}` {direction} than `{baseline_label}` for `{parameter}`. "
        f"The RMSE is `{result.rmse:.2f} {units}`, which describes the typical grid-cell difference."
    )


def _overall_analysis_text(table: pd.DataFrame) -> str:
    if table.empty:
        return "No products available for analysis."
    highest_tec = table.loc[table["TEC max"].idxmax()]
    lowest_muf = table.loc[table["MUF min"].idxmin()]
    return (
        f"**Automated analysis:** the largest TEC value appears in `{highest_tec['Product']}` "
        f"at `{highest_tec['TEC max']:.2f} TECU`. The lowest MUF3000F2 appears in "
        f"`{lowest_muf['Product']}` at `{lowest_muf['MUF min']:.2f} MHz`. These two indicators "
        "support separate GNSS-positioning and HF-communication risk checks rather than relying on TEC alone."
    )


if __name__ == "__main__":
    main()

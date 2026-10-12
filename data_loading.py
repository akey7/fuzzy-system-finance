"""
Load and validate the files the backend delivers to the S3 buckets.

Every loader in this module follows INTERFACE_CONTRACT.md (contract version 7)
and raises DataLoadError with a human-readable message when a file is
missing, does not parse, predates contract v4, or lacks a required key,
column, group, or dataset. Anything the contract does not require is ignored,
so additive backend changes do not break the frontend.

This module does no downloading and no UI construction.
"""

import math
import os
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import yaml

CONTRACT_VERSION = 7

OPTIMIZATION_METADATA_KEY = "optimization_metadata.yml"
PLOT_DATA_KEY = "portfolio_optimization_plot_data.h5"
FORECASTS_KEY = "all_forecasts.csv"

SUPPORTED_ANNUALIZATION = "arithmetic"
SUPPORTED_UNITS = "percent_per_year"
PLOT_GROUPS = (
    "efficient_frontier",
    "tangency_line",
    "simulated_portfolios",
    "max_sharpe_ratio",
    "min_var_portfolio",
)
ARIMA_SUFFIX = "_arima"
WEIGHT_ZERO_THRESHOLD = 1e-6


class DataLoadError(Exception):
    """
    Raised when a backend file is missing, malformed, or does not satisfy
    the interface contract. The message is suitable for display on the UI.
    """


@dataclass(frozen=True)
class PortfolioPoint:
    """
    Annualized (arithmetic) return and risk of a single portfolio, both in
    percent.
    """

    annualized_return: float
    risk: float


@dataclass(frozen=True)
class OptimizationMetadata:
    """
    Contents of optimization_metadata.yml that the frontend uses.

    Returns, risks, and the risk-free rate are annualized and in percent.
    Weights are in percent and are exactly as delivered (including
    floating-point noise); use clean_weights() before display.
    """

    date_from: str
    date_to: str
    tickers: list[str]
    risk_free_rate: float
    risk_free_rate_date: str
    weights: dict[str, float]
    optimum: PortfolioPoint
    sharpe_ratio: float
    min_var: PortfolioPoint
    annualization_method: str
    periods_per_year: int


@dataclass(frozen=True)
class XYData:
    """
    One group from portfolio_optimization_plot_data.h5. xs is annualized
    risk (σ, %) and ys is annualized return (%). xs may contain NaN.
    """

    xs: np.ndarray
    ys: np.ndarray


def _require_file(path: str, key: str) -> None:
    """
    Raise DataLoadError if a local copy of a bucket file does not exist.

    Parameters
    ----------
    path : str
        Local path of the downloaded file.

    key : str
        Bucket key of the file, used in the error message.
    """
    if not os.path.isfile(path):
        raise DataLoadError(f"`{key}` is not available.")


def _get(mapping: dict, dotted_path: str):
    """
    Look up a nested key in a parsed YAML mapping.

    Parameters
    ----------
    mapping : dict
        Parsed YAML document.

    dotted_path : str
        Path such as "optimum_portfolio.weights".

    Returns
    -------
    Any
        The value at that path.
    """
    node = mapping
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise DataLoadError(
                f"`{OPTIMIZATION_METADATA_KEY}` is missing required key `{dotted_path}`."
            )
        node = node[key]
    return node


def _get_float(mapping: dict, dotted_path: str) -> float:
    """
    Look up a nested key in a parsed YAML mapping and convert it to float.

    Parameters
    ----------
    mapping : dict
        Parsed YAML document.

    dotted_path : str
        Path such as "optimum_portfolio.risk".

    Returns
    -------
    float
        The value at that path.
    """
    value = _get(mapping, dotted_path)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise DataLoadError(
            f"`{OPTIMIZATION_METADATA_KEY}` key `{dotted_path}` is not a number."
        )


def load_optimization_metadata(path: str) -> OptimizationMetadata:
    """
    Load and validate optimization_metadata.yml.

    Files without an `annualization` key predate contract v4 (compounded
    return, no Sharpe ratio or minimum variance portfolio) and are rejected.

    Parameters
    ----------
    path : str
        Local path of the downloaded file.

    Returns
    -------
    OptimizationMetadata
        Validated metadata.
    """
    _require_file(path, OPTIMIZATION_METADATA_KEY)
    try:
        with open(path, "r", encoding="utf-8") as file:
            document = yaml.safe_load(file)
    except (yaml.YAMLError, UnicodeDecodeError):
        raise DataLoadError(f"`{OPTIMIZATION_METADATA_KEY}` is not valid YAML.")
    if not isinstance(document, dict):
        raise DataLoadError(f"`{OPTIMIZATION_METADATA_KEY}` is not a YAML mapping.")
    if "annualization" not in document:
        raise DataLoadError(
            f"`{OPTIMIZATION_METADATA_KEY}` predates interface contract v4 "
            "(no `annualization` key). Rerun the backend to produce current data."
        )
    method = str(_get(document, "annualization.method"))
    if method != SUPPORTED_ANNUALIZATION:
        raise DataLoadError(
            f"`{OPTIMIZATION_METADATA_KEY}` uses unsupported annualization "
            f"method `{method}`."
        )

    tickers = _get(document, "tickers")
    weights = _get(document, "optimum_portfolio.weights")
    if not isinstance(tickers, list) or not tickers:
        raise DataLoadError(f"`{OPTIMIZATION_METADATA_KEY}` key `tickers` is empty.")
    if not isinstance(weights, dict) or not weights:
        raise DataLoadError(
            f"`{OPTIMIZATION_METADATA_KEY}` key `optimum_portfolio.weights` is empty."
        )
    try:
        weights = {str(ticker): float(weight) for ticker, weight in weights.items()}
    except (TypeError, ValueError):
        raise DataLoadError(
            f"`{OPTIMIZATION_METADATA_KEY}` has a non-numeric portfolio weight."
        )

    return OptimizationMetadata(
        date_from=str(_get(document, "date_updated.date_from")),
        date_to=str(_get(document, "date_updated.date_to")),
        tickers=[str(ticker) for ticker in tickers],
        risk_free_rate=_get_float(document, "risk_free_rate.risk_free_rate"),
        risk_free_rate_date=str(_get(document, "risk_free_rate.risk_free_rate_date"))[
            :10
        ],
        weights=weights,
        optimum=PortfolioPoint(
            annualized_return=_get_float(
                document, "optimum_portfolio.annualized_return"
            ),
            risk=_get_float(document, "optimum_portfolio.risk"),
        ),
        sharpe_ratio=_get_float(document, "optimum_portfolio.sharpe_ratio"),
        min_var=PortfolioPoint(
            annualized_return=_get_float(
                document, "min_var_portfolio.annualized_return"
            ),
            risk=_get_float(document, "min_var_portfolio.risk"),
        ),
        annualization_method=method,
        periods_per_year=int(_get_float(document, "annualization.periods_per_year")),
    )


def clean_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    Replace floating-point noise in portfolio weights with exact zeros.

    Weights with absolute value below WEIGHT_ZERO_THRESHOLD (in percent)
    are set to 0.0. Other weights are returned unchanged; round them for
    display. The weights are not renormalized, so they sum to 100 only
    within the optimizer's tolerance.

    Parameters
    ----------
    weights : dict[str, float]
        Ticker to weight in percent.

    Returns
    -------
    dict[str, float]
        Ticker to cleaned weight in percent.
    """
    return {
        ticker: 0.0 if abs(weight) < WEIGHT_ZERO_THRESHOLD else weight
        for ticker, weight in weights.items()
    }


def _attr_str(value) -> str:
    """
    Convert an HDF5 string attribute (str or bytes) to str.

    Parameters
    ----------
    value : str | bytes
        Attribute value read with h5py.

    Returns
    -------
    str
        Decoded attribute value.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_plot_data(path: str) -> dict[str, XYData]:
    """
    Load and validate portfolio_optimization_plot_data.h5.

    Files without a root `annualization` attribute predate contract v4
    (daily values) and are rejected, as are files whose units are not
    annualized percent. Array lengths are read from the file.

    Parameters
    ----------
    path : str
        Local path of the downloaded file.

    Returns
    -------
    dict[str, XYData]
        Group name (see PLOT_GROUPS) to its xs and ys arrays.
    """
    _require_file(path, PLOT_DATA_KEY)
    try:
        with h5py.File(path, "r") as hf:
            if "annualization" not in hf.attrs:
                raise DataLoadError(
                    f"`{PLOT_DATA_KEY}` predates interface contract v4 (daily "
                    "values, no `annualization` attribute). Rerun the backend to "
                    "produce current data."
                )
            method = _attr_str(hf.attrs["annualization"])
            if method != SUPPORTED_ANNUALIZATION:
                raise DataLoadError(
                    f"`{PLOT_DATA_KEY}` uses unsupported annualization method "
                    f"`{method}`."
                )
            for attr in ("return_units", "risk_units"):
                units = _attr_str(hf.attrs.get(attr, ""))
                if units != SUPPORTED_UNITS:
                    raise DataLoadError(
                        f"`{PLOT_DATA_KEY}` attribute `{attr}` is `{units}`, "
                        f"expected `{SUPPORTED_UNITS}`."
                    )
            plot_data = {}
            for group in PLOT_GROUPS:
                for dataset in ("xs", "ys"):
                    if f"{group}/{dataset}" not in hf:
                        raise DataLoadError(
                            f"`{PLOT_DATA_KEY}` is missing dataset `{group}/{dataset}`."
                        )
                xs = np.asarray(hf[f"{group}/xs"][:], dtype=float)
                ys = np.asarray(hf[f"{group}/ys"][:], dtype=float)
                if xs.ndim != 1 or xs.shape != ys.shape or xs.size == 0:
                    raise DataLoadError(
                        f"`{PLOT_DATA_KEY}` group `{group}` must hold two non-empty "
                        "one-dimensional arrays of equal length."
                    )
                plot_data[group] = XYData(xs=xs, ys=ys)
    except OSError:
        raise DataLoadError(f"`{PLOT_DATA_KEY}` is not a readable HDF5 file.")
    return plot_data


def optimization_files_agree(
    metadata: OptimizationMetadata,
    plot_data: dict[str, XYData],
    rel_tol: float = 1e-6,
) -> bool:
    """
    Check that the YAML and HDF5 files come from the same backend run.

    The backend uploads the two files one after the other, so for a few
    seconds they can come from different runs. Within one run, the
    max Sharpe ratio and minimum variance points in the HDF5 file equal
    the YAML values.

    Parameters
    ----------
    metadata : OptimizationMetadata
        Loaded optimization_metadata.yml.

    plot_data : dict[str, XYData]
        Loaded portfolio_optimization_plot_data.h5.

    rel_tol : float
        Relative tolerance for the comparison.

    Returns
    -------
    bool
        True if both points match.
    """
    pairs = [
        (plot_data["max_sharpe_ratio"], metadata.optimum),
        (plot_data["min_var_portfolio"], metadata.min_var),
    ]
    return all(
        math.isclose(xy.xs[0], point.risk, rel_tol=rel_tol)
        and math.isclose(xy.ys[0], point.annualized_return, rel_tol=rel_tol)
        for xy, point in pairs
    )


def forecast_tickers(columns) -> list[str]:
    """
    Find the tickers in all_forecasts.csv.

    A ticker T is present when the file has both a `T_arima` forecast
    column and a `T` actual-price column. Other columns are ignored.

    Parameters
    ----------
    columns : Iterable[str]
        Column names of the forecasts DataFrame.

    Returns
    -------
    list[str]
        Sorted tickers.
    """
    names = {str(column) for column in columns}
    return sorted(
        name[: -len(ARIMA_SUFFIX)]
        for name in names
        if name.endswith(ARIMA_SUFFIX) and name[: -len(ARIMA_SUFFIX)] in names
    )


def load_forecasts(path: str) -> pd.DataFrame:
    """
    Load and validate all_forecasts.csv.

    The index is `pred_date`, normalized to dates (the time of day in the
    file has no meaning) and sorted ascending. Actual-price columns are
    NaN on rows with no actual value yet (the live forecast row).

    Parameters
    ----------
    path : str
        Local path of the downloaded file.

    Returns
    -------
    pd.DataFrame
        Forecasts indexed by date, with numeric `T` and `T_arima` columns
        for each ticker returned by forecast_tickers().
    """
    _require_file(path, FORECASTS_KEY)
    try:
        df = pd.read_csv(path, index_col="pred_date", encoding="utf-8")
        df.index = pd.to_datetime(df.index).normalize()
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise DataLoadError(f"`{FORECASTS_KEY}` is not a readable CSV file.")
    except ValueError:
        raise DataLoadError(f"`{FORECASTS_KEY}` needs a `pred_date` column of dates.")
    tickers = forecast_tickers(df.columns)
    if not tickers:
        raise DataLoadError(
            f"`{FORECASTS_KEY}` has no ticker with both `T` and `T_arima` columns."
        )
    for ticker in tickers:
        for column in (ticker, f"{ticker}{ARIMA_SUFFIX}"):
            try:
                df[column] = pd.to_numeric(df[column], errors="raise")
            except (TypeError, ValueError):
                raise DataLoadError(
                    f"`{FORECASTS_KEY}` column `{column}` is not numeric."
                )
    df.index.name = "pred_date"
    return df.sort_index()


def last_actual_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """
    Find the latest date with at least one actual price.

    Parameters
    ----------
    df : pd.DataFrame
        Forecasts from load_forecasts().

    Returns
    -------
    pd.Timestamp | None
        Latest date with actual data, or None if there is none.
    """
    actuals = df[forecast_tickers(df.columns)]
    dates = actuals.index[actuals.notna().any(axis=1)]
    return dates.max() if len(dates) else None


def live_forecast_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """
    Find the date of the live (not yet happened) forecast.

    Per the contract, the final row is the first business day after the
    latest data and has no actual prices.

    Parameters
    ----------
    df : pd.DataFrame
        Forecasts from load_forecasts().

    Returns
    -------
    pd.Timestamp | None
        Date of the final row if it has no actual prices, else None.
    """
    if df.empty:
        return None
    final_actuals = df[forecast_tickers(df.columns)].iloc[-1]
    return df.index[-1] if final_actuals.isna().all() else None

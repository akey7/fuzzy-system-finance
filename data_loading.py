"""
Load and validate the files the backend delivers to the S3 buckets.

Every loader in this module follows docs/INTERFACE_CONTRACT.md (contract version 30)
and raises DataLoadError with a human-readable message when a file is
missing, does not parse, predates contract v4, or lacks a required key,
column, group, or dataset. Anything the contract does not require is ignored,
so additive backend changes do not break the frontend.

All files live in one bucket (a DigitalOcean Space) under two top-level
folders, at keys `<folder>/YYYY-MM-DD/<file>`. This module also finds the
available dates from folder listings whose keys are relative to the folder
(the `<folder>/` prefix already removed, as S3Downloader.list_keys returns
them).

This module does no downloading and no UI construction.
"""

import datetime
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import yaml

CONTRACT_VERSION = 30

OPTIMIZATION_METADATA_KEY = "optimization_metadata.yml"
PLOT_DATA_KEY = "portfolio_optimization_plot_data.h5"
FORECASTS_KEY = "all_forecasts.csv"
# Optional (added in contract v21); not part of the date availability rule.
FORECASTS_METADATA_KEY = "all_forecasts_metadata.yml"

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
# Optional all_forecasts.csv columns: model orders (v16, `_arima_d` since v18)
# and the naive baseline (v19). Matched by exact suffix.
ORDER_SUFFIXES = ("_arima_p", "_arima_d", "_arima_q")
NAIVE_SUFFIX = "_naive"
# Units all_forecasts_metadata.yml must declare for the metrics it reports.
EXPECTED_METRIC_UNITS = {
    "rmse": "usd_per_share",
    "mae": "usd_per_share",
    "mape": "percent",
    "relative_mae": "ratio",
    "skill": "ratio",
    "directional_accuracy": "fraction",
}
WEIGHT_ZERO_THRESHOLD = 1e-6
DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})/", re.ASCII)


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
    floating-point noise); use clean_weights() before display. `weights`
    belongs to the maximum Sharpe ratio portfolio (`optimum_portfolio` in
    the file) and `min_var_weights` to the minimum variance portfolio
    (contract v28+, None for older files).
    """

    date_from: str
    date_to: str
    tickers: list[str]
    risk_free_rate: float
    risk_free_rate_date: str
    weights: dict[str, float]
    min_var_weights: dict[str, float] | None
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


@dataclass(frozen=True)
class ModelMetrics:
    """
    Accuracy of one forecast model (ARIMA or naive) for one ticker, from
    all_forecasts_metadata.yml. Any value can be None (null in the file).

    rmse and mae are in US dollars per share, mape in percent, and
    directional_accuracy a fraction from 0 to 1 (ARIMA only).
    """

    rmse: float | None
    mae: float | None
    mape: float | None
    directional_accuracy: float | None = None
    n_directional: int | None = None


@dataclass(frozen=True)
class TickerMetrics:
    """
    ARIMA and naive accuracy for one ticker over its evaluation window,
    from all_forecasts_metadata.yml. relative_mae is ARIMA MAE / naive MAE
    (below 1 means ARIMA was more accurate) and skill is 1 - relative_mae
    (above 0 means ARIMA beat the naive forecast). Any value can be None.
    """

    n_evaluated: int | None
    evaluation_start_date: str | None
    evaluation_end_date: str | None
    arima: ModelMetrics
    naive: ModelMetrics
    relative_mae: float | None
    skill: float | None


@dataclass(frozen=True)
class ForecastsMetadata:
    """
    Contents of all_forecasts_metadata.yml (contract v21) that the frontend
    uses. Ticker keys in ticker_metrics are lowercase, as in the file; look
    tickers up with metrics_for(), which ignores case. The ARIMA order
    selection rule and its tie-break are optional free text from the
    backend (the tie-break since v23), None when absent.
    """

    training_window_start_date: str
    training_window_end_date: str
    forecast_date: str
    training_window_returns: int
    horizon_trading_days: int
    target_trading_days: int
    evaluation_start_date: str | None
    evaluation_end_date: str | None
    arima_description: str | None
    naive_description: str | None
    arima_order_selection: str | None
    arima_order_selection_tie_break: str | None
    ticker_metrics: dict[str, TickerMetrics]

    def metrics_for(self, ticker: str) -> TickerMetrics | None:
        """
        Return the metrics for a ticker, matching case-insensitively.

        Parameters
        ----------
        ticker : str
            Ticker in any case, e.g. "XBI".

        Returns
        -------
        TickerMetrics | None
            The ticker's metrics, or None if the file has none for it.
        """
        return self.ticker_metrics.get(ticker.lower())


def key_date(key: str) -> str | None:
    """
    Return the date prefix of a key relative to its top-level folder.

    Parameters
    ----------
    key : str
        Key with the `<folder>/` prefix removed, such as
        "2026-10-09/all_forecasts.csv".

    Returns
    -------
    str | None
        The `YYYY-MM-DD` date, or None if the key does not start with a
        valid date prefix (including impossible dates such as 2026-13-40
        and stale objects left directly under the folder by contract v8
        and earlier).
    """
    match = DATE_PREFIX_RE.match(key)
    if match is None:
        return None
    try:
        datetime.date.fromisoformat(match.group(1))
    except ValueError:
        return None
    return match.group(1)


def marker_dates(keys: Iterable[str], filename: str) -> set[str]:
    """
    Find the dates whose folder contains a given file.

    Only keys of exactly the form `YYYY-MM-DD/<filename>` count; keys
    without a valid date prefix or nested deeper are ignored.

    Parameters
    ----------
    keys : Iterable[str]
        Every key listed from one top-level folder, relative to it.

    filename : str
        File name to look for, such as OPTIMIZATION_METADATA_KEY.

    Returns
    -------
    set[str]
        Dates (`YYYY-MM-DD`) that contain the file.
    """
    dates = set()
    for key in keys:
        date = key_date(key)
        if date is not None and key == f"{date}/{filename}":
            dates.add(date)
    return dates


def available_dates(
    portfolio_keys: Iterable[str], forecast_keys: Iterable[str]
) -> list[str]:
    """
    Find the dates that are complete in both top-level folders.

    Per the contract, a date is available only when the portfolio folder
    has `YYYY-MM-DD/optimization_metadata.yml` (uploaded last, so it marks
    a complete date) and the time series folder has
    `YYYY-MM-DD/all_forecasts.csv`. Dates need not be consecutive.

    Parameters
    ----------
    portfolio_keys : Iterable[str]
        Every key in the FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER folder, relative
        to the folder.

    forecast_keys : Iterable[str]
        Every key in the FSF_FRONT_END_TIME_SERIES_FOLDER folder, relative to the
        folder.

    Returns
    -------
    list[str]
        Available dates, newest first.
    """
    dates = marker_dates(portfolio_keys, OPTIMIZATION_METADATA_KEY) & marker_dates(
        forecast_keys, FORECASTS_KEY
    )
    return sorted(dates, reverse=True)


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


def _read_yaml_mapping(path: str, file_key: str) -> dict:
    """
    Read a YAML file that must contain a mapping at the top level.

    Parameters
    ----------
    path : str
        Local path of the downloaded file.

    file_key : str
        File name, used in error messages.

    Returns
    -------
    dict
        Parsed YAML document.
    """
    _require_file(path, file_key)
    try:
        with open(path, "r", encoding="utf-8") as file:
            document = yaml.safe_load(file)
    except (yaml.YAMLError, UnicodeDecodeError):
        raise DataLoadError(f"`{file_key}` is not valid YAML.")
    if not isinstance(document, dict):
        raise DataLoadError(f"`{file_key}` is not a YAML mapping.")
    return document


def _get(mapping: dict, dotted_path: str, file_key: str = OPTIMIZATION_METADATA_KEY):
    """
    Look up a nested key in a parsed YAML mapping.

    Parameters
    ----------
    mapping : dict
        Parsed YAML document.

    dotted_path : str
        Path such as "optimum_portfolio.weights".

    file_key : str
        File name, used in error messages.

    Returns
    -------
    Any
        The value at that path.
    """
    node = mapping
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise DataLoadError(
                f"`{file_key}` is missing required key `{dotted_path}`."
            )
        node = node[key]
    return node


def _get_float(
    mapping: dict, dotted_path: str, file_key: str = OPTIMIZATION_METADATA_KEY
) -> float:
    """
    Look up a nested key in a parsed YAML mapping and convert it to float.

    Parameters
    ----------
    mapping : dict
        Parsed YAML document.

    dotted_path : str
        Path such as "optimum_portfolio.risk".

    file_key : str
        File name, used in error messages.

    Returns
    -------
    float
        The value at that path.
    """
    value = _get(mapping, dotted_path, file_key)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise DataLoadError(f"`{file_key}` key `{dotted_path}` is not a number.")


def _parse_weights(node, dotted_path: str) -> dict[str, float]:
    """
    Validate a portfolio weight map from optimization_metadata.yml.

    Parameters
    ----------
    node : Any
        The parsed `weights` mapping (ticker to weight in percent).

    dotted_path : str
        Path of the map, used in error messages.

    Returns
    -------
    dict[str, float]
        Ticker to weight in percent, exactly as delivered (floating-point
        noise included; see clean_weights()).
    """
    if not isinstance(node, dict) or not node:
        raise DataLoadError(
            f"`{OPTIMIZATION_METADATA_KEY}` key `{dotted_path}` is empty."
        )
    try:
        return {str(ticker): float(weight) for ticker, weight in node.items()}
    except (TypeError, ValueError):
        raise DataLoadError(
            f"`{OPTIMIZATION_METADATA_KEY}` key `{dotted_path}` has a non-numeric weight."
        )


def load_optimization_metadata(path: str) -> OptimizationMetadata:
    """
    Load and validate optimization_metadata.yml.

    Files without an `annualization` key predate contract v4 (compounded
    return, no Sharpe ratio or minimum variance portfolio) and are rejected.
    The minimum variance portfolio's weights (contract v28) are optional:
    they are None when absent (older files) and, if malformed, ignored
    with a console message so the rest of the file can still be shown.

    Parameters
    ----------
    path : str
        Local path of the downloaded file.

    Returns
    -------
    OptimizationMetadata
        Validated metadata.
    """
    document = _read_yaml_mapping(path, OPTIMIZATION_METADATA_KEY)
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
    if not isinstance(tickers, list) or not tickers:
        raise DataLoadError(f"`{OPTIMIZATION_METADATA_KEY}` key `tickers` is empty.")
    weights = _parse_weights(
        _get(document, "optimum_portfolio.weights"), "optimum_portfolio.weights"
    )
    min_var_weights = None
    min_var_node = document.get("min_var_portfolio")
    if isinstance(min_var_node, dict) and "weights" in min_var_node:
        try:
            min_var_weights = _parse_weights(
                min_var_node["weights"], "min_var_portfolio.weights"
            )
        except DataLoadError as error:
            print(f"Ignoring optional minimum variance weights: {error}")

    return OptimizationMetadata(
        date_from=str(_get(document, "date_updated.date_from")),
        date_to=str(_get(document, "date_updated.date_to")),
        tickers=[str(ticker) for ticker in tickers],
        risk_free_rate=_get_float(document, "risk_free_rate.risk_free_rate"),
        risk_free_rate_date=str(_get(document, "risk_free_rate.risk_free_rate_date"))[
            :10
        ],
        weights=weights,
        min_var_weights=min_var_weights,
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


def _clean_optional_column(df: pd.DataFrame, column: str, whole_numbers: bool) -> None:
    """
    Convert an optional all_forecasts.csv column to numbers in place.

    Empty cells become missing values. If any non-empty cell is not a
    number (or, for whole_numbers, not a whole number), the column is
    dropped so the rest of the file can still be used.

    Parameters
    ----------
    df : pd.DataFrame
        Forecasts being loaded.

    column : str
        Column name; nothing happens if it is absent.

    whole_numbers : bool
        True for model order columns, which become nullable integers
        (pandas "Int64"); False for float columns.
    """
    if column not in df.columns:
        return
    original = df[column]
    converted = pd.to_numeric(original, errors="coerce")
    malformed = converted.isna() & original.notna()
    if whole_numbers:
        malformed |= converted.notna() & (converted % 1 != 0)
    if malformed.any():
        print(f"Ignoring malformed optional column {column} in {FORECASTS_KEY}.")
        df.drop(columns=column, inplace=True)
        return
    df[column] = converted.astype("Int64") if whole_numbers else converted.astype(float)


def arima_orders(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Return the ARIMA model orders behind each of a ticker's forecasts.

    `p`, `d`, and `q` describe ARIMA(p, d, q) on log prices (the model is
    fit as ARIMA(p, 0, q) on daily log returns). Orders are missing on rows
    with no forecast and for columns the date folder does not have (no
    order columns before contract v16, no `d` before v18).

    Parameters
    ----------
    df : pd.DataFrame
        Forecasts from load_forecasts().

    ticker : str
        The ticker.

    Returns
    -------
    pd.DataFrame
        Columns `p`, `d`, `q` (nullable integers), indexed like df.
    """
    orders = pd.DataFrame(index=df.index)
    for name, suffix in zip(("p", "d", "q"), ORDER_SUFFIXES):
        column = f"{ticker}{suffix}"
        if column in df.columns:
            orders[name] = df[column]
        else:
            orders[name] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    return orders


def naive_column(df: pd.DataFrame, ticker: str) -> str | None:
    """
    Return the name of a ticker's naive-forecast column, if present.

    Parameters
    ----------
    df : pd.DataFrame
        Forecasts from load_forecasts().

    ticker : str
        The ticker.

    Returns
    -------
    str | None
        `T_naive`, or None if the date folder predates contract v19.
    """
    column = f"{ticker}{NAIVE_SUFFIX}"
    return column if column in df.columns else None


def load_forecasts(path: str) -> pd.DataFrame:
    """
    Load and validate all_forecasts.csv.

    The index is `pred_date`, normalized to dates (the time of day in the
    file has no meaning) and sorted ascending. Actual-price columns are
    NaN on rows with no actual value yet (the live forecast row), and
    `T_arima` can be NaN where the previous close was missing (v20).

    Optional columns are converted when present: `T_naive` (v19) to
    float, and `T_arima_p`, `T_arima_d`, `T_arima_q` (v16, v18) to
    nullable integers. A malformed optional column is dropped rather than
    rejecting the file.

    Parameters
    ----------
    path : str
        Local path of the downloaded file.

    Returns
    -------
    pd.DataFrame
        Forecasts indexed by date, with numeric `T` and `T_arima` columns
        for each ticker returned by forecast_tickers(), plus any optional
        columns present.
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
        _clean_optional_column(df, f"{ticker}{NAIVE_SUFFIX}", whole_numbers=False)
        for suffix in ORDER_SUFFIXES:
            _clean_optional_column(df, f"{ticker}{suffix}", whole_numbers=True)
    df.index.name = "pred_date"
    return df.sort_index()


def _optional_str(node, key: str) -> str | None:
    """
    Return a mapping value as a string, or None if missing or null.

    Parameters
    ----------
    node : Any
        A parsed YAML mapping (anything else counts as missing).

    key : str
        Key to look up.

    Returns
    -------
    str | None
        The value as a string, or None.
    """
    if not isinstance(node, dict) or node.get(key) is None:
        return None
    return str(node[key])


def _optional_number(
    node, key: str, dotted_path: str, integer: bool = False
) -> float | int | None:
    """
    Return a mapping value as a number, or None if missing or null.

    Parameters
    ----------
    node : Any
        A parsed YAML mapping (anything else counts as missing).

    key : str
        Key to look up.

    dotted_path : str
        Full path of the key, used in the error message.

    integer : bool
        Return an int instead of a float.

    Returns
    -------
    float | int | None
        The value, or None.
    """
    if not isinstance(node, dict) or node.get(key) is None:
        return None
    try:
        return int(node[key]) if integer else float(node[key])
    except (TypeError, ValueError):
        raise DataLoadError(
            f"`{FORECASTS_METADATA_KEY}` key `{dotted_path}` is not a number."
        )


def _model_metrics(node, dotted_path: str) -> ModelMetrics:
    """
    Read one model's metrics block from all_forecasts_metadata.yml.

    Parameters
    ----------
    node : Any
        The `arima` or `naive` mapping (missing counts as all-null).

    dotted_path : str
        Path of the block, used in error messages.

    Returns
    -------
    ModelMetrics
        The metrics, with None for anything missing or null.
    """
    return ModelMetrics(
        rmse=_optional_number(node, "rmse", f"{dotted_path}.rmse"),
        mae=_optional_number(node, "mae", f"{dotted_path}.mae"),
        mape=_optional_number(node, "mape", f"{dotted_path}.mape"),
        directional_accuracy=_optional_number(
            node, "directional_accuracy", f"{dotted_path}.directional_accuracy"
        ),
        n_directional=_optional_number(
            node, "n_directional", f"{dotted_path}.n_directional", integer=True
        ),
    )


def load_forecasts_metadata(path: str) -> ForecastsMetadata:
    """
    Load and validate all_forecasts_metadata.yml (contract v21).

    The `units` block must declare the units the contract specifies for
    every metric, so values are never shown with the wrong units. Metric
    values may be null, and per-ticker keys are lowercase in the file.
    Unknown keys are ignored.

    Parameters
    ----------
    path : str
        Local path of the downloaded file.

    Returns
    -------
    ForecastsMetadata
        Validated metadata.
    """
    key = FORECASTS_METADATA_KEY
    document = _read_yaml_mapping(path, key)
    for metric, expected in EXPECTED_METRIC_UNITS.items():
        found = str(_get(document, f"units.{metric}", key))
        if found != expected:
            raise DataLoadError(
                f"`{key}` declares `{metric}` in `{found}`, expected `{expected}`."
            )
    ticker_nodes = _get(document, "ticker_metrics", key)
    if not isinstance(ticker_nodes, dict):
        raise DataLoadError(f"`{key}` key `ticker_metrics` is not a mapping.")
    metrics = {}
    for ticker, node in ticker_nodes.items():
        prefix = f"ticker_metrics.{ticker}"
        if not isinstance(node, dict):
            raise DataLoadError(f"`{key}` key `{prefix}` is not a mapping.")
        metrics[str(ticker).lower()] = TickerMetrics(
            n_evaluated=_optional_number(
                node, "n_evaluated", f"{prefix}.n_evaluated", integer=True
            ),
            evaluation_start_date=_optional_str(node, "evaluation_start_date"),
            evaluation_end_date=_optional_str(node, "evaluation_end_date"),
            arima=_model_metrics(node.get("arima"), f"{prefix}.arima"),
            naive=_model_metrics(node.get("naive"), f"{prefix}.naive"),
            relative_mae=_optional_number(
                node, "relative_mae", f"{prefix}.relative_mae"
            ),
            skill=_optional_number(node, "skill", f"{prefix}.skill"),
        )
    evaluation_window = document.get("evaluation_window")
    models = document.get("models")
    models = models if isinstance(models, dict) else {}
    return ForecastsMetadata(
        training_window_start_date=str(
            _get(document, "forecast_window.training_window_start_date", key)
        ),
        training_window_end_date=str(
            _get(document, "forecast_window.training_window_end_date", key)
        ),
        forecast_date=str(_get(document, "forecast_window.forecast_date", key)),
        training_window_returns=int(
            _get_float(document, "forecast_window.training_window_returns", key)
        ),
        horizon_trading_days=int(
            _get_float(document, "forecast_window.horizon_trading_days", key)
        ),
        target_trading_days=int(
            _get_float(document, "evaluation_window.target_trading_days", key)
        ),
        evaluation_start_date=_optional_str(evaluation_window, "start_date"),
        evaluation_end_date=_optional_str(evaluation_window, "end_date"),
        arima_description=_optional_str(models.get("arima"), "description"),
        naive_description=_optional_str(models.get("naive"), "description"),
        arima_order_selection=_optional_str(models.get("arima"), "order_selection"),
        arima_order_selection_tie_break=_optional_str(
            models.get("arima"), "order_selection_tie_break"
        ),
        ticker_metrics=metrics,
    )


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


def folder_date_warnings(
    date: str,
    metadata: OptimizationMetadata | None,
    forecasts: pd.DataFrame | None,
    forecasts_metadata: ForecastsMetadata | None = None,
) -> list[str]:
    """
    Check that the files in a date folder describe that date.

    Per the contract, the folder date equals `date_updated.date_to` in the
    metadata, the last date with an actual price in the forecasts, and
    `forecast_window.training_window_end_date` in the forecasts metadata,
    whose `forecast_window.forecast_date` equals the forecasts' live
    forecast date. Files that are not available are skipped.

    Parameters
    ----------
    date : str
        The folder date (`YYYY-MM-DD`).

    metadata : OptimizationMetadata | None
        Loaded optimization_metadata.yml from that folder, if available.

    forecasts : pd.DataFrame | None
        Loaded all_forecasts.csv from that folder, if available.

    forecasts_metadata : ForecastsMetadata | None
        Loaded all_forecasts_metadata.yml from that folder, if available.

    Returns
    -------
    list[str]
        Human-readable warnings; empty if everything agrees.
    """
    warnings = []
    if metadata is not None and metadata.date_to != date:
        warnings.append(
            f"`{OPTIMIZATION_METADATA_KEY}` in folder {date} has data through "
            f"{metadata.date_to}."
        )
    if forecasts is not None:
        as_of = last_actual_date(forecasts)
        as_of_text = as_of.date().isoformat() if as_of is not None else None
        if as_of_text != date:
            warnings.append(
                f"`{FORECASTS_KEY}` in folder {date} has actual prices through "
                f"{as_of_text or 'no date'}."
            )
    if forecasts_metadata is not None:
        if forecasts_metadata.training_window_end_date != date:
            warnings.append(
                f"`{FORECASTS_METADATA_KEY}` in folder {date} has a training "
                f"window ending {forecasts_metadata.training_window_end_date}."
            )
        live_date = live_forecast_date(forecasts) if forecasts is not None else None
        if (
            live_date is not None
            and live_date.date().isoformat() != forecasts_metadata.forecast_date
        ):
            warnings.append(
                f"`{FORECASTS_METADATA_KEY}` in folder {date} describes a forecast "
                f"for {forecasts_metadata.forecast_date}, but `{FORECASTS_KEY}` "
                f"forecasts {live_date.date().isoformat()}."
            )
    return warnings

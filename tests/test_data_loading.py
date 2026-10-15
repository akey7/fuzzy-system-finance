"""
Tests for data_loading.py against synthetic files that follow
INTERFACE_CONTRACT.md. All files are written to pytest's tmp_path;
nothing under input/ is read or written.
"""

import math

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from data_loading import (
    PLOT_GROUPS,
    DataLoadError,
    arima_orders,
    available_dates,
    clean_weights,
    folder_date_warnings,
    forecast_tickers,
    key_date,
    load_forecasts_metadata,
    marker_dates,
    naive_column,
    last_actual_date,
    live_forecast_date,
    load_forecasts,
    load_optimization_metadata,
    load_plot_data,
    optimization_files_agree,
)


def v7_metadata() -> dict:
    """
    Return a contract v7 optimization_metadata.yml document, based on the
    contract's example, plus an unknown key that readers must ignore.
    """
    return {
        "annualization": {"method": "arithmetic", "periods_per_year": 252},
        "date_updated": {"date_from": "2025-10-07", "date_to": "2026-10-09"},
        "min_var_portfolio": {"annualized_return": 12.85, "risk": 6.89},
        "optimum_portfolio": {
            "annualized_return": 34.34,
            "risk": 12.44,
            "sharpe_ratio": 2.435,
            "weights": {
                "FXE": 3.055998668072217e-14,
                "QQQ": 60.00000000000006,
                "XLE": 39.99999999999994,
            },
        },
        "risk_free_rate": {
            "daily_risk_free_rate": 0.01607142857142857,
            "risk_free_rate": 4.05,
            "risk_free_rate_date": "2026-10-08 00:00:00",
        },
        "tickers": ["XLE", "QQQ", "FXE"],
        "some_future_key": {"ignored": True},
    }


def write_yaml(path, document) -> str:
    """
    Write a YAML document and return its path as a string.
    """
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(document, file)
    return str(path)


def write_h5(path, attrs=None, skip_dataset=None) -> str:
    """
    Write a contract v7 plot data file and return its path as a string.

    Parameters
    ----------
    attrs : dict | None
        Root attributes. Defaults to the contract v7 attributes.

    skip_dataset : str | None
        A "group/dataset" path to leave out.
    """
    if attrs is None:
        attrs = {
            "return_units": "percent_per_year",
            "risk_units": "percent_per_year",
            "return_type": "simple",
            "annualization": "arithmetic",
            "periods_per_year": np.int64(252),
        }
    points = {
        "efficient_frontier": ([6.89, np.nan, 13.0], [12.85, 20.0, 35.0]),
        "tangency_line": ([0.0, 6.0, 12.44], [4.05, 18.66, 34.34]),
        "simulated_portfolios": ([7.0, 8.0, 9.0, 10.0], [10.0, 11.0, 12.0, 13.0]),
        "max_sharpe_ratio": ([12.44], [34.34]),
        "min_var_portfolio": ([6.89], [12.85]),
    }
    with h5py.File(path, "w") as hf:
        for key, value in attrs.items():
            hf.attrs[key] = value
        for group, (xs, ys) in points.items():
            for name, values in (("xs", xs), ("ys", ys)):
                if f"{group}/{name}" != skip_dataset:
                    hf.create_dataset(f"{group}/{name}", data=np.array(values))
        hf.create_dataset("unknown_group/xs", data=np.array([1.0]))
    return str(path)


def write_csv(path, text) -> str:
    """
    Write CSV text and return its path as a string.
    """
    path.write_text(text, encoding="utf-8")
    return str(path)


FORECASTS_CSV = (
    "pred_date,GLD_arima,GLD,NVDA_arima,NVDA,extra_column\n"
    "2026-10-07 17:00:00,300.0,301.0,180.0,182.0,x\n"
    "2026-10-08 17:00:00,302.0,300.0,181.0,179.0,y\n"
    "2026-10-09 17:00:00,301.0,,180.5,,z\n"
)


# optimization_metadata.yml


def test_metadata_loads_v7_fields(tmp_path):
    path = write_yaml(tmp_path / "m.yml", v7_metadata())
    metadata = load_optimization_metadata(path)
    assert metadata.date_to == "2026-10-09"
    assert metadata.risk_free_rate == 4.05
    assert metadata.risk_free_rate_date == "2026-10-08"
    assert metadata.sharpe_ratio == 2.435
    assert metadata.min_var.risk == 6.89
    assert metadata.periods_per_year == 252
    assert set(metadata.tickers) == {"XLE", "QQQ", "FXE"}


def test_metadata_rejects_pre_v4(tmp_path):
    document = v7_metadata()
    del document["annualization"]
    path = write_yaml(tmp_path / "m.yml", document)
    with pytest.raises(DataLoadError, match="predates interface contract v4"):
        load_optimization_metadata(path)


def test_metadata_reports_missing_key_path(tmp_path):
    document = v7_metadata()
    del document["optimum_portfolio"]["sharpe_ratio"]
    path = write_yaml(tmp_path / "m.yml", document)
    with pytest.raises(DataLoadError, match="optimum_portfolio.sharpe_ratio"):
        load_optimization_metadata(path)


def test_metadata_missing_file(tmp_path):
    with pytest.raises(DataLoadError, match="not available"):
        load_optimization_metadata(str(tmp_path / "missing.yml"))


def test_metadata_invalid_yaml(tmp_path):
    path = tmp_path / "m.yml"
    path.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(DataLoadError, match="not valid YAML"):
        load_optimization_metadata(str(path))


def test_clean_weights_zeroes_noise_only():
    cleaned = clean_weights(
        {"FXE": 3.055998668072217e-14, "XLE": 39.99999999999994, "NEG": -1e-9}
    )
    assert cleaned["FXE"] == 0.0
    assert cleaned["NEG"] == 0.0
    assert cleaned["XLE"] == 39.99999999999994


# portfolio_optimization_plot_data.h5


def test_plot_data_loads_all_groups(tmp_path):
    plot_data = load_plot_data(write_h5(tmp_path / "p.h5"))
    assert set(plot_data) == set(PLOT_GROUPS)
    assert plot_data["simulated_portfolios"].xs.shape == (4,)
    assert math.isnan(plot_data["efficient_frontier"].xs[1])


def test_plot_data_rejects_pre_v4(tmp_path):
    path = write_h5(tmp_path / "p.h5", attrs={"some_other_attr": "x"})
    with pytest.raises(DataLoadError, match="predates interface contract v4"):
        load_plot_data(path)


def test_plot_data_rejects_wrong_units(tmp_path):
    attrs = {
        "annualization": "arithmetic",
        "return_units": "percent_per_day",
        "risk_units": "percent_per_year",
    }
    path = write_h5(tmp_path / "p.h5", attrs=attrs)
    with pytest.raises(DataLoadError, match="return_units"):
        load_plot_data(path)


def test_plot_data_reports_missing_dataset(tmp_path):
    path = write_h5(tmp_path / "p.h5", skip_dataset="tangency_line/ys")
    with pytest.raises(DataLoadError, match="tangency_line/ys"):
        load_plot_data(path)


def test_plot_data_not_hdf5(tmp_path):
    path = tmp_path / "p.h5"
    path.write_text("not hdf5", encoding="utf-8")
    with pytest.raises(DataLoadError, match="not a readable HDF5"):
        load_plot_data(str(path))


def test_optimization_files_agree(tmp_path):
    metadata = load_optimization_metadata(write_yaml(tmp_path / "m.yml", v7_metadata()))
    plot_data = load_plot_data(write_h5(tmp_path / "p.h5"))
    assert optimization_files_agree(metadata, plot_data)

    document = v7_metadata()
    document["optimum_portfolio"]["risk"] = 15.0
    other = load_optimization_metadata(write_yaml(tmp_path / "m2.yml", document))
    assert not optimization_files_agree(other, plot_data)


# all_forecasts.csv


def test_forecast_tickers_ignores_unpaired_columns():
    columns = ["GLD_arima", "GLD", "NVDA_arima", "QQQ", "extra_column"]
    assert forecast_tickers(columns) == ["GLD"]


def test_forecasts_load_and_dates(tmp_path):
    df = load_forecasts(write_csv(tmp_path / "f.csv", FORECASTS_CSV))
    assert forecast_tickers(df.columns) == ["GLD", "NVDA"]
    assert df.index[0] == pd.Timestamp("2026-10-07")
    assert pd.isna(df.loc[pd.Timestamp("2026-10-09"), "GLD"])
    assert last_actual_date(df) == pd.Timestamp("2026-10-08")
    assert live_forecast_date(df) == pd.Timestamp("2026-10-09")


def test_forecasts_no_live_row(tmp_path):
    text = FORECASTS_CSV.replace("301.0,,180.5,,z", "301.0,299.0,180.5,181.0,z")
    df = load_forecasts(write_csv(tmp_path / "f.csv", text))
    assert live_forecast_date(df) is None
    assert last_actual_date(df) == pd.Timestamp("2026-10-09")


def test_forecasts_missing_pred_date(tmp_path):
    path = write_csv(tmp_path / "f.csv", "date,GLD_arima,GLD\n2026-10-07,1,2\n")
    with pytest.raises(DataLoadError, match="pred_date"):
        load_forecasts(path)


def test_forecasts_without_ticker_pairs(tmp_path):
    path = write_csv(tmp_path / "f.csv", "pred_date,GLD\n2026-10-07,1\n")
    with pytest.raises(DataLoadError, match="no ticker"):
        load_forecasts(path)


def test_forecasts_non_numeric_column(tmp_path):
    path = write_csv(
        tmp_path / "f.csv", "pred_date,GLD_arima,GLD\n2026-10-07,abc,2\n"
    )
    with pytest.raises(DataLoadError, match="GLD_arima"):
        load_forecasts(path)


# Date prefixes and availability


@pytest.mark.parametrize(
    "key, expected",
    [
        ("2026-10-09/all_forecasts.csv", "2026-10-09"),
        ("2026-10-09/", "2026-10-09"),
        ("all_forecasts.csv", None),  # stale root-level object (contract v8 and earlier)
        ("2026-13-40/all_forecasts.csv", None),  # impossible date
        ("2026-1-9/all_forecasts.csv", None),  # not zero-padded
        ("x2026-10-09/all_forecasts.csv", None),
        ("2026-10-09all_forecasts.csv", None),
    ],
)
def test_key_date(key, expected):
    assert key_date(key) == expected


def test_marker_dates_requires_exact_key():
    keys = [
        "2026-10-09/optimization_metadata.yml",
        "2026-10-08/portfolio_optimization_plot_data.h5",  # no marker
        "2026-10-07/nested/optimization_metadata.yml",  # nested too deep
        "optimization_metadata.yml",  # stale root-level object
        "2026-02-30/optimization_metadata.yml",  # impossible date
    ]
    assert marker_dates(keys, "optimization_metadata.yml") == {"2026-10-09"}


def test_available_dates_needs_both_buckets_newest_first():
    portfolio_keys = [
        "2026-10-09/optimization_metadata.yml",
        "2026-10-09/portfolio_optimization_plot_data.h5",
        "2026-10-06/optimization_metadata.yml",
        "2026-09-30/optimization_metadata.yml",
        "2026-10-02/portfolio_optimization_plot_data.h5",  # .yml not uploaded yet
        "optimization_metadata.yml",
    ]
    forecast_keys = [
        "2026-10-09/all_forecasts.csv",
        "2026-10-06/all_forecasts.csv",
        "2026-10-02/all_forecasts.csv",
        "2026-10-01/all_forecasts.csv",  # forecasts only
        "all_forecasts.csv",
    ]
    assert available_dates(portfolio_keys, forecast_keys) == [
        "2026-10-09",
        "2026-10-06",
    ]


def test_available_dates_empty():
    assert available_dates([], ["2026-10-09/all_forecasts.csv"]) == []


def test_folder_date_warnings(tmp_path):
    metadata = load_optimization_metadata(write_yaml(tmp_path / "m.yml", v7_metadata()))
    forecasts = load_forecasts(write_csv(tmp_path / "f.csv", FORECASTS_CSV))
    # v7_metadata() has date_to 2026-10-09; FORECASTS_CSV has actuals through 2026-10-08.
    assert folder_date_warnings("2026-10-09", metadata, None) == []
    assert folder_date_warnings("2026-10-08", None, forecasts) == []
    assert folder_date_warnings("2026-10-09", None, None) == []
    warnings = folder_date_warnings("2026-10-09", metadata, forecasts)
    assert len(warnings) == 1
    assert "all_forecasts.csv" in warnings[0]
    assert len(folder_date_warnings("2026-10-07", metadata, forecasts)) == 2


# all_forecasts.csv, contract v16-v21 columns

# Six columns per ticker. NVDA's close is treated as missing before
# 2026-10-08, so that row has an empty forecast and empty orders (v20),
# while its naive forecast is still present. The last row is the live one.
V21_FORECASTS_CSV = (
    "pred_date,GLD_arima,GLD_arima_p,GLD_arima_d,GLD_arima_q,GLD_naive,GLD,"
    "NVDA_arima,NVDA_arima_p,NVDA_arima_d,NVDA_arima_q,NVDA_naive,NVDA\n"
    "2026-10-07 17:00:00,300.0,1,1,0,299.0,301.0,180.0,0,1,1,181.0,182.0\n"
    "2026-10-08 17:00:00,302.0,1,1,1,301.0,300.0,,,,,182.0,179.0\n"
    "2026-10-09 17:00:00,301.0,0,1,1,300.0,,180.5,1,1,0,179.0,\n"
)


def test_v21_forecasts_tickers_by_exact_suffix(tmp_path):
    df = load_forecasts(write_csv(tmp_path / "f.csv", V21_FORECASTS_CSV))
    # Order and naive columns must not be mistaken for tickers or forecasts.
    assert forecast_tickers(df.columns) == ["GLD", "NVDA"]
    assert last_actual_date(df) == pd.Timestamp("2026-10-08")
    assert live_forecast_date(df) == pd.Timestamp("2026-10-09")


def test_v21_order_columns_are_nullable_integers(tmp_path):
    df = load_forecasts(write_csv(tmp_path / "f.csv", V21_FORECASTS_CSV))
    orders = arima_orders(df, "NVDA")
    assert list(orders.columns) == ["p", "d", "q"]
    assert str(orders["p"].dtype) == "Int64"
    first = orders.loc[pd.Timestamp("2026-10-07")]
    assert (first["p"], first["d"], first["q"]) == (0, 1, 1)
    assert orders.loc[pd.Timestamp("2026-10-08")].isna().all()
    assert pd.isna(df.loc[pd.Timestamp("2026-10-08"), "NVDA_arima"])


def test_v21_naive_column(tmp_path):
    df = load_forecasts(write_csv(tmp_path / "f.csv", V21_FORECASTS_CSV))
    assert naive_column(df, "GLD") == "GLD_naive"
    assert df["NVDA_naive"].notna().all()


def test_older_forecasts_have_no_optional_columns(tmp_path):
    df = load_forecasts(write_csv(tmp_path / "f.csv", FORECASTS_CSV))
    assert naive_column(df, "GLD") is None
    assert arima_orders(df, "GLD").isna().all().all()


def test_pre_v18_orders_without_d(tmp_path):
    text = "pred_date,GLD_arima,GLD_arima_p,GLD_arima_q,GLD\n2026-10-07,300,1,0,301\n"
    df = load_forecasts(write_csv(tmp_path / "f.csv", text))
    orders = arima_orders(df, "GLD").iloc[0]
    assert orders["p"] == 1 and orders["q"] == 0 and pd.isna(orders["d"])


def test_malformed_optional_columns_are_dropped(tmp_path):
    text = (
        "pred_date,GLD_arima,GLD_arima_p,GLD_naive,GLD\n"
        "2026-10-07,300,1.5,abc,301\n"
    )
    df = load_forecasts(write_csv(tmp_path / "f.csv", text))
    # Non-whole order and non-numeric naive value: both ignored, file still loads.
    assert "GLD_arima_p" not in df.columns
    assert "GLD_naive" not in df.columns
    assert forecast_tickers(df.columns) == ["GLD"]


# all_forecasts_metadata.yml (contract v21)


def v21_forecasts_metadata() -> dict:
    """
    Return an all_forecasts_metadata.yml document based on the contract's
    example, with lowercase ticker keys, a null metric, and an unknown key.
    """
    return {
        "evaluation_window": {
            "end_date": "2026-10-08",
            "start_date": "2026-09-10",
            "target_trading_days": 20,
        },
        "forecast_window": {
            "forecast_date": "2026-10-09",
            "horizon_trading_days": 1,
            "training_window_end_date": "2026-10-08",
            "training_window_returns": 20,
            "training_window_start_date": "2026-09-09",
        },
        "models": {
            "arima": {"d": 1, "description": "ARIMA(p, 0, q) on daily log returns"},
            "naive": {"description": "the most recent observed close (random walk)"},
        },
        "ticker_metrics": {
            "gld": {
                "arima": {
                    "directional_accuracy": 0.55,
                    "mae": 1.21,
                    "mape": 1.32,
                    "n_directional": 20,
                    "rmse": 1.52,
                },
                "evaluation_end_date": "2026-10-08",
                "evaluation_start_date": "2026-09-10",
                "n_evaluated": 20,
                "naive": {"mae": 1.25, "mape": 1.37, "rmse": 1.58},
                "relative_mae": 0.968,
                "skill": 0.032,
            },
            "nvda": {
                "arima": {"directional_accuracy": None, "mae": 2.0, "n_directional": 0},
                "n_evaluated": 19,
                "naive": {"mae": 2.5},
                "relative_mae": 0.8,
                "skill": None,
            },
        },
        "units": {
            "directional_accuracy": "fraction",
            "mae": "usd_per_share",
            "mape": "percent",
            "relative_mae": "ratio",
            "rmse": "usd_per_share",
            "skill": "ratio",
        },
        "some_future_key": [1, 2, 3],
    }


def test_forecasts_metadata_loads(tmp_path):
    fm = load_forecasts_metadata(write_yaml(tmp_path / "fm.yml", v21_forecasts_metadata()))
    assert fm.forecast_date == "2026-10-09"
    assert fm.training_window_end_date == "2026-10-08"
    assert fm.training_window_returns == 20
    assert fm.target_trading_days == 20
    gld = fm.metrics_for("GLD")  # uppercase lookup of a lowercase key
    assert gld.arima.rmse == 1.52
    assert gld.naive.mape == 1.37
    assert gld.arima.n_directional == 20
    assert gld.skill == 0.032


def test_forecasts_metadata_nulls_and_missing_values(tmp_path):
    fm = load_forecasts_metadata(write_yaml(tmp_path / "fm.yml", v21_forecasts_metadata()))
    nvda = fm.metrics_for("nvda")
    assert nvda.arima.directional_accuracy is None
    assert nvda.skill is None
    assert nvda.arima.rmse is None  # key absent
    assert nvda.evaluation_start_date is None
    assert fm.metrics_for("QQQ") is None


def test_forecasts_metadata_rejects_wrong_units(tmp_path):
    document = v21_forecasts_metadata()
    document["units"]["mape"] = "fraction"
    path = write_yaml(tmp_path / "fm.yml", document)
    with pytest.raises(DataLoadError, match="mape"):
        load_forecasts_metadata(path)


def test_forecasts_metadata_requires_ticker_metrics(tmp_path):
    document = v21_forecasts_metadata()
    del document["ticker_metrics"]
    path = write_yaml(tmp_path / "fm.yml", document)
    with pytest.raises(DataLoadError, match="ticker_metrics"):
        load_forecasts_metadata(path)


def test_forecasts_metadata_non_numeric_metric(tmp_path):
    document = v21_forecasts_metadata()
    document["ticker_metrics"]["gld"]["arima"]["rmse"] = "big"
    path = write_yaml(tmp_path / "fm.yml", document)
    with pytest.raises(DataLoadError, match="ticker_metrics.gld.arima.rmse"):
        load_forecasts_metadata(path)


def test_forecasts_metadata_missing_file(tmp_path):
    with pytest.raises(DataLoadError, match="not available"):
        load_forecasts_metadata(str(tmp_path / "missing.yml"))


def test_folder_date_warnings_forecasts_metadata(tmp_path):
    forecasts = load_forecasts(write_csv(tmp_path / "f.csv", FORECASTS_CSV))
    fm = load_forecasts_metadata(write_yaml(tmp_path / "fm.yml", v21_forecasts_metadata()))
    # Folder 2026-10-08: actuals through 10-08, training window ends 10-08,
    # and the live forecast is for 10-09 in both files.
    assert folder_date_warnings("2026-10-08", None, forecasts, fm) == []

    document = v21_forecasts_metadata()
    document["forecast_window"]["forecast_date"] = "2026-10-10"
    document["forecast_window"]["training_window_end_date"] = "2026-10-07"
    other = load_forecasts_metadata(write_yaml(tmp_path / "fm2.yml", document))
    warnings = folder_date_warnings("2026-10-08", None, forecasts, other)
    assert len(warnings) == 2
    assert all("all_forecasts_metadata.yml" in warning for warning in warnings)

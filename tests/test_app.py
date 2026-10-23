"""
Tests for the text helpers in app.py that describe how the forecasts are
made and scored and how ARIMA compared with the naive forecast, and for
the date conversion behind the time series plot. No Gradio app is built,
and nothing under input/ or the network is touched.
"""

import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

from app import (
    accuracy_verdict,
    metrics_windows_text,
    noon_timestamps,
    order_selection_text,
    training_window_text,
)


def test_noon_timestamps_keep_calendar_date():
    # Closes as load_forecasts() gives them (midnight), plus one with a
    # time of day, out of order and with a non-default index.
    # All in one format: pandas infers the format from the first string.
    dates = pd.Series(
        pd.to_datetime(["2026-10-09 00:00", "2026-10-08 17:00", "2026-10-12 00:00"]),
        index=[5, 3, 9],
    )
    result = noon_timestamps(dates)
    assert list(result) == list(
        pd.to_datetime(["2026-10-09 12:00", "2026-10-08 12:00", "2026-10-12 12:00"])
    )
    assert list(result.index) == [5, 3, 9]


def test_noon_timestamps_raise_no_warnings():
    # Adding a pd.Timedelta here raised a NumPy DeprecationWarning (generic
    # timedelta unit) that a future NumPy will turn into an error.
    dates = pd.Series(pd.to_datetime(["2026-10-08", "2026-10-09"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        noon_timestamps(dates)


def test_training_window_text_fills_closes():
    # Closes = training_window_returns + 1, whatever the backend setting.
    assert training_window_text(251) == (
        "a window of the most recent closes before that day (252 trading days)"
    )
    assert training_window_text(20) == (
        "a window of the most recent closes before that day (21 trading days)"
    )


def test_training_window_text_without_metadata():
    # Before contract v21: no metadata, so the parenthesis is left out.
    assert training_window_text(None) == (
        "a window of the most recent closes before that day"
    )


def test_metrics_windows_text():
    assert metrics_windows_text(251, 60) == (
        "ARIMA training windows are 252 trading days in length. 60 trading days "
        "is a modest sample, and consecutive forecasts are not independent, so "
        "the effective sample is smaller still; treat these as indicative."
    )


def test_metrics_windows_text_other_settings():
    # Older dates: 20 returns (21 closes) of training and 20 scored days.
    text = metrics_windows_text(20, 20)
    assert text.startswith("ARIMA training windows are 21 trading days in length.")
    assert "20 trading days is a modest sample" in text


@pytest.mark.parametrize("relative_mae", [0.981, 0.995, 1.0, 1.019])
def test_accuracy_verdict_approximately_tied(relative_mae):
    assert "**approximately tied**" in accuracy_verdict(relative_mae)


@pytest.mark.parametrize("relative_mae", [0.5, 0.98])
def test_accuracy_verdict_smaller(relative_mae):
    # 0.98 itself is outside the open band.
    assert "**smaller**" in accuracy_verdict(relative_mae)


@pytest.mark.parametrize("relative_mae", [1.02, 1.5])
def test_accuracy_verdict_larger(relative_mae):
    # 1.02 itself is outside the open band.
    assert "**larger**" in accuracy_verdict(relative_mae)


def test_accuracy_verdict_missing():
    assert accuracy_verdict(None).startswith("Not enough scored days")


def test_order_selection_text_with_tie_break():
    fm = SimpleNamespace(
        arima_order_selection="lowest BIC",
        arima_order_selection_tie_break="fewer parameters (p + q), then lower p",
    )
    assert order_selection_text(fm) == (
        "Orders are chosen by lowest BIC. "
        "Ties go to fewer parameters (p + q), then lower p."
    )


def test_order_selection_text_without_tie_break():
    fm = SimpleNamespace(
        arima_order_selection="lowest BIC", arima_order_selection_tie_break=None
    )
    assert order_selection_text(fm) == "Orders are chosen by lowest BIC."


def test_order_selection_text_absent():
    fm = SimpleNamespace(arima_order_selection=None, arima_order_selection_tie_break=None)
    assert order_selection_text(fm) == ""
    assert order_selection_text(None) == ""

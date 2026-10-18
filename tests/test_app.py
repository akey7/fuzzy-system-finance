"""
Tests for the text helpers in app.py that describe how the forecasts are
made and scored. No Gradio app is built, and nothing under input/ or the network is
touched.
"""

from types import SimpleNamespace

from app import metrics_windows_text, order_selection_text, training_window_text


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

"""
Tests for the text helpers in app.py that describe how the forecasts are
made and scored. No Gradio app is built, and nothing under input/ or the network is
touched.
"""

from types import SimpleNamespace

from app import metrics_windows_text, order_selection_text, training_window_text


def test_training_window_text_one_trading_year():
    # Contract v24: 252 closes, 251 daily log returns.
    assert training_window_text(251) == "the previous trading year (252 closes)"


def test_training_window_text_older_dates():
    # Before contract v24: 20 daily log returns.
    assert training_window_text(20) == "the previous 21 closes (20 daily log returns)"


def test_training_window_text_without_metadata():
    assert training_window_text(None) == "a short window of recent trading days"


def test_metrics_windows_text_one_trading_year():
    # Contract v24: 251 returns (252 closes) of training, 20 scored days.
    assert metrics_windows_text(251, 20) == (
        "ARIMA training windows are 252 trading days in length. Accuracy is "
        "scored over only the last 20 trading days, which is a small sample, "
        "so treat these figures as indicative."
    )


def test_metrics_windows_text_older_dates():
    # Before contract v24: 20 returns (21 closes) of training.
    text = metrics_windows_text(20, 20)
    assert text.startswith("ARIMA training windows are 21 trading days in length.")
    assert "last 20 trading days" in text


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

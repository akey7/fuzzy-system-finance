"""
Tests for the text helpers in app.py that describe how the forecasts are
made. No Gradio app is built, and nothing under input/ or the network is
touched.
"""

from types import SimpleNamespace

from app import order_selection_text, training_window_text


def test_training_window_text_one_trading_year():
    # Contract v24: 252 closes, 251 daily log returns.
    assert training_window_text(251) == "the previous trading year (252 closes)"


def test_training_window_text_older_dates():
    # Before contract v24: 20 daily log returns.
    assert training_window_text(20) == "the previous 21 closes (20 daily log returns)"


def test_training_window_text_without_metadata():
    assert training_window_text(None) == "a short window of recent trading days"


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

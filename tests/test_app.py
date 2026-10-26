"""
Tests for the text helpers in app.py that describe how the forecasts are
made and scored and how ARIMA compared with the naive forecast, for the
date conversion behind the time series plot, for the portfolio weights
table, for the portfolio optimization plot (capital allocation line
clipping, NaN frontier points, legend placement, labels, and the maximum
return marker), and for the text and tables built from the contract v32
portfolio sections (assets, constraints, random portfolios). No Gradio
app is built, and nothing under input/ or the network is touched.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from app import (
    PORTFOLIO_PLOT_SUBTITLE,
    PORTFOLIO_PLOT_TITLE,
    RANDOM_PORTFOLIOS_CAPTION,
    accuracy_verdict,
    capital_allocation_line,
    metrics_windows_text,
    noon_timestamps,
    order_selection_text,
    portfolio_optimization_figure,
    portfolio_plot_subtitle,
    portfolio_weights_df,
    training_window_text,
)
from app import (
    assets_caption_markdown,
    assets_df,
    constraints_markdown,
    near_frontier_sentence,
    portfolio_caption_markdown,
    random_portfolios_details_markdown,
)
from data_loading import (
    MaxReturnPortfolio,
    PortfolioConstraints,
    PortfolioPoint,
    RandomPortfolioStats,
    XYData,
)

RISK_FREE_RATE = 4.04
SHARPE_RATIO = 2.44
MAX_SHARPE_X = 12.44
MAX_SHARPE_Y = RISK_FREE_RATE + SHARPE_RATIO * MAX_SHARPE_X


def plot_data(cal_end_x, frontier_xs=None):
    """
    Build synthetic plot data. The capital allocation line runs from 0 to
    cal_end_x: MAX_SHARPE_X for a v31 file, further for an older file whose
    line overshoots the max Sharpe ratio point.
    """
    cal_xs = np.linspace(0.0, cal_end_x, 100)
    if frontier_xs is None:
        frontier_xs = np.linspace(7.0, cal_end_x, 20)
    frontier_xs = np.asarray(frontier_xs, dtype=float)
    return {
        "efficient_frontier": XYData(
            xs=frontier_xs, ys=np.linspace(12.0, 40.0, frontier_xs.size)
        ),
        "tangency_line": XYData(xs=cal_xs, ys=RISK_FREE_RATE + SHARPE_RATIO * cal_xs),
        "simulated_portfolios": XYData(
            xs=np.array([9.0, 11.0, 14.0]), ys=np.array([15.0, 20.0, 25.0])
        ),
        "max_sharpe_ratio": XYData(
            xs=np.array([MAX_SHARPE_X]), ys=np.array([MAX_SHARPE_Y])
        ),
        "min_var_portfolio": XYData(xs=np.array([7.0]), ys=np.array([12.0])),
    }


METADATA = SimpleNamespace(risk_free_rate=RISK_FREE_RATE, periods_per_year=252)


def title_text(ax):
    """Return the title annotation drawn above the subtitle."""
    matches = [text for text in ax.texts if text.get_text() == PORTFOLIO_PLOT_TITLE]
    assert len(matches) == 1
    return matches[0]


def line_by_label(ax, prefix):
    """Return the plotted line whose legend label starts with prefix."""
    matches = [line for line in ax.lines if line.get_label().startswith(prefix)]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("cal_end_x", [MAX_SHARPE_X, 30.0], ids=["v31", "pre_v31"])
def test_capital_allocation_line_ends_at_max_sharpe_point(cal_end_x):
    data = plot_data(cal_end_x)
    cal = capital_allocation_line(data["tangency_line"], data["max_sharpe_ratio"])
    # Starts at (0, risk-free rate), stops exactly at the max Sharpe point.
    assert cal.xs[0] == 0.0 and cal.ys[0] == pytest.approx(RISK_FREE_RATE)
    assert cal.xs[-1] == MAX_SHARPE_X and cal.ys[-1] == MAX_SHARPE_Y
    assert np.all(cal.xs <= MAX_SHARPE_X)
    # No duplicate end point and no point added beyond the file's own.
    assert np.all(np.diff(cal.xs) > 0)
    assert cal.xs.size <= data["tangency_line"].xs.size


def test_capital_allocation_line_treats_near_end_as_end():
    # A last point a hair past the max Sharpe risk (floating-point noise) is
    # replaced by the exact max Sharpe point, not kept beside it.
    xs = np.array([0.0, 6.0, MAX_SHARPE_X * (1 + 1e-12)])
    line = XYData(xs=xs, ys=RISK_FREE_RATE + SHARPE_RATIO * xs)
    point = XYData(xs=np.array([MAX_SHARPE_X]), ys=np.array([MAX_SHARPE_Y]))
    cal = capital_allocation_line(line, point)
    assert list(cal.xs) == [0.0, 6.0, MAX_SHARPE_X]
    assert cal.ys[-1] == MAX_SHARPE_Y


@pytest.mark.parametrize("cal_end_x", [MAX_SHARPE_X, 30.0], ids=["v31", "pre_v31"])
def test_plot_draws_clipped_capital_allocation_line(cal_end_x):
    fig = portfolio_optimization_figure(plot_data(cal_end_x), METADATA)
    ax = fig.axes[0]
    line = line_by_label(ax, "Capital allocation line")
    assert line.get_label() == "Capital allocation line (risk-free rate 4.04%)"
    assert line.get_xdata()[-1] == MAX_SHARPE_X
    assert line.get_ydata()[-1] == MAX_SHARPE_Y
    assert max(line.get_xdata()) <= MAX_SHARPE_X


def test_plot_limits_leave_no_room_for_old_line():
    # The pre-v31 line would reach 4.04 + 2.44 * 30 = 77.2% return; the
    # highest plotted point after clipping is the frontier's 40%.
    fig = portfolio_optimization_figure(plot_data(30.0), METADATA)
    ax = fig.axes[0]
    assert ax.get_ylim()[1] < 45.0
    assert ax.get_xlim()[1] < 32.0
    assert ax.get_xlim()[0] <= 0.0


def test_plot_breaks_frontier_at_nan():
    frontier_xs = [7.0, 8.0, np.nan, 10.0, 11.0, np.nan]
    fig = portfolio_optimization_figure(
        plot_data(MAX_SHARPE_X, frontier_xs), METADATA
    )
    ax = fig.axes[0]
    xdata = np.asarray(line_by_label(ax, "Efficient frontier").get_xdata(), dtype=float)
    # NaN stays in the line, which matplotlib draws as a break.
    assert list(np.isnan(xdata)) == [False, False, True, False, False, True]
    assert all(np.isfinite(ax.get_xlim())) and all(np.isfinite(ax.get_ylim()))


def test_plot_legend_outside_axes_and_not_clipped():
    fig = portfolio_optimization_figure(plot_data(30.0), METADATA)
    # Gradio saves the figure without tight bounding boxes, so check the
    # layout as drawn in a plain render.
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    ax = fig.axes[0]
    axes_box = ax.get_window_extent(renderer)
    legend_box = ax.get_legend().get_window_extent(renderer)
    figure_box = fig.bbox
    # Right of the axes, not over the data.
    assert legend_box.x0 >= axes_box.x1
    # Inside the saved image.
    assert legend_box.x1 <= figure_box.x1
    assert legend_box.y0 >= figure_box.y0 and legend_box.y1 <= figure_box.y1
    title_box = title_text(ax).get_window_extent(renderer)
    subtitle_box = ax._left_title.get_window_extent(renderer)
    for box in (
        title_box,
        subtitle_box,
        ax.xaxis.label.get_window_extent(renderer),
        ax.yaxis.label.get_window_extent(renderer),
    ):
        assert box.x0 >= figure_box.x0 and box.x1 <= figure_box.x1
        assert box.y0 >= figure_box.y0 and box.y1 <= figure_box.y1


def test_plot_area_square_and_titles_aligned_with_y_axis():
    fig = portfolio_optimization_figure(plot_data(30.0), METADATA)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    ax = fig.axes[0]
    axes_box = ax.get_window_extent(renderer)
    # Square plot area (to within a pixel of rounding).
    assert axes_box.width == pytest.approx(axes_box.height, abs=1.0)
    # Title and subtitle start at the y-axis, title above the subtitle.
    title_box = title_text(ax).get_window_extent(renderer)
    subtitle_box = ax._left_title.get_window_extent(renderer)
    assert title_box.x0 == pytest.approx(axes_box.x0, abs=2.0)
    assert subtitle_box.x0 == pytest.approx(axes_box.x0, abs=2.0)
    assert title_box.y0 >= subtitle_box.y1
    assert subtitle_box.y0 >= axes_box.y1


def test_plot_titles_axes_and_legend_labels():
    fig = portfolio_optimization_figure(plot_data(MAX_SHARPE_X), METADATA)
    ax = fig.axes[0]
    assert title_text(ax).get_text() == "Mean–Variance Portfolio Optimization"
    # Two lines, split at the semicolon; no data date on the figure.
    assert ax.get_title(loc="left") == (
        "Expected returns and volatility estimated from daily simple returns;\n"
        "annualized using 252 trading days."
    )
    assert "Data as of" not in ax.get_title(loc="left")
    assert ax.get_xlabel() == "Annualized Volatility (σ, %)"
    assert ax.get_ylabel() == "Annualized Arithmetic Expected Return (%)"
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == [
        "Efficient frontier",
        "Capital allocation line (risk-free rate 4.04%)",
        "Random portfolios",
        "Max Sharpe ratio portfolio",
        "Minimum variance portfolio",
    ]


def test_plot_legend_keys_opaque_but_points_keep_transparency():
    fig = portfolio_optimization_figure(plot_data(MAX_SHARPE_X), METADATA)
    ax = fig.axes[0]
    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    key = legend.legend_handles[labels.index("Random portfolios")]
    assert key.get_alpha() == 1.0
    cloud = [c for c in ax.collections if c.get_label() == "Random portfolios"]
    assert len(cloud) == 1 and cloud[0].get_alpha() == pytest.approx(0.1)


def test_plot_without_metadata():
    # Fail soft: plot still drawn, with the fallback subtitle.
    fig = portfolio_optimization_figure(plot_data(MAX_SHARPE_X), None)
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == PORTFOLIO_PLOT_SUBTITLE
    assert line_by_label(ax, "Capital allocation line").get_label() == (
        "Capital allocation line"
    )


def test_portfolio_plot_subtitle():
    assert portfolio_plot_subtitle(None) == (
        "Expected returns and volatility estimated from daily simple returns;\n"
        "annualized using 252 trading days."
    )
    assert portfolio_plot_subtitle(260).endswith("annualized using 260 trading days.")


def test_random_portfolios_caption_has_no_disclaimer():
    assert RANDOM_PORTFOLIOS_CAPTION == (
        "Random portfolios are sampled from the same constraints as the optimizer "
        "to illustrate the feasible region; they are not evidence that the frontier "
        "is correct."
    )
    assert "investment" not in RANDOM_PORTFOLIOS_CAPTION.lower()


def test_portfolio_weights_df_both_portfolios():
    metadata = SimpleNamespace(
        weights={"FXE": 3.1e-14, "QQQ": 60.00000000000006, "XLE": 39.99999999999994},
        min_var_weights={"FXE": 39.99999999999998, "QQQ": 1.2e-15, "XLE": 60.0},
        max_return=None,
    )
    df = portfolio_weights_df(metadata)
    assert list(df.columns) == ["Ticker", "Max Sharpe Weight", "Min Variance Weight"]
    # Sorted by max Sharpe weight, then minimum variance weight; noise shown as 0.
    assert df.values.tolist() == [
        ["QQQ", "60.00%", "0.00%"],
        ["XLE", "40.00%", "60.00%"],
        ["FXE", "0.00%", "40.00%"],
    ]


def test_portfolio_weights_df_before_v28():
    # No minimum variance weights: no column for them.
    metadata = SimpleNamespace(
        weights={"QQQ": 60.0, "XLE": 40.0}, min_var_weights=None, max_return=None
    )
    df = portfolio_weights_df(metadata)
    assert list(df.columns) == ["Ticker", "Max Sharpe Weight"]
    assert df.values.tolist() == [["QQQ", "60.00%"], ["XLE", "40.00%"]]


def test_portfolio_weights_df_ticker_in_one_portfolio_only():
    metadata = SimpleNamespace(
        weights={"QQQ": 100.0},
        min_var_weights={"QQQ": 70.0, "GLD": 30.0},
        max_return=None,
    )
    df = portfolio_weights_df(metadata)
    assert df.values.tolist() == [
        ["QQQ", "100.00%", "70.00%"],
        ["GLD", "n/a", "30.00%"],
    ]


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
def test_accuracy_verdict_within_two_percent(relative_mae):
    assert accuracy_verdict(relative_mae) == (
        "ARIMA's average error was **within 2%** of the naive forecast's."
    )


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


# Contract v32 portfolio sections


def random_stats(share=0.0004, tolerance=1.0):
    """Random portfolio statistics like the contract's v32 example."""
    return RandomPortfolioStats(
        count=10000,
        max_annualized_return=32.2,
        mean_annualized_return=11.3,
        equal_weight_annualized_return=11.2,
        return_std=5.9,
        return_std_uniform_without_cap=6.4,
        near_frontier_tolerance=tolerance,
        near_frontier_share=share,
    )


MAX_RETURN = MaxReturnPortfolio(
    point=PortfolioPoint(annualized_return=50.0, risk=35.0),
    weights={"FXE": 1e-15, "QQQ": 60.0, "XLE": 39.99999999999998},
)
CAPPED = PortfolioConstraints(
    long_only=True, fully_invested=True, weight_cap=40.00000000000001
)
UNCAPPED = PortfolioConstraints(long_only=True, fully_invested=True, weight_cap=100.0)


def v32_namespace(**overrides):
    """Metadata with every v32 section, for the app's text helpers."""
    values = dict(
        weights={"FXE": 3.1e-14, "QQQ": 60.00000000000006, "XLE": 39.99999999999994},
        min_var_weights={"FXE": 39.99999999999998, "QQQ": 1.2e-15, "XLE": 60.0},
        max_return=MAX_RETURN,
        assets={
            "FXE": PortfolioPoint(annualized_return=-2.104, risk=7.4567),
            "XLE": PortfolioPoint(annualized_return=38.9, risk=19.4),
            "QQQ": PortfolioPoint(annualized_return=30.2, risk=18.0),
        },
        constraints=CAPPED,
        random_portfolios=random_stats(),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_portfolio_weights_df_with_max_return():
    df = portfolio_weights_df(v32_namespace())
    assert list(df.columns) == [
        "Ticker",
        "Max Sharpe Weight",
        "Min Variance Weight",
        "Max Return Weight",
    ]
    # Noise (1e-15) shown as zero, and the cap's 39.99999999999998 rounded.
    assert df.values.tolist() == [
        ["QQQ", "60.00%", "0.00%", "60.00%"],
        ["XLE", "40.00%", "60.00%", "40.00%"],
        ["FXE", "0.00%", "40.00%", "0.00%"],
    ]


def test_assets_df_numeric_and_sorted_by_return():
    df = assets_df(v32_namespace())
    assert list(df.columns) == [
        "Ticker",
        "Annualized return (%)",
        "Annualized volatility (σ, %)",
    ]
    assert df.values.tolist() == [
        ["XLE", 38.9, 19.4],
        ["QQQ", 30.2, 18.0],
        ["FXE", -2.1, 7.46],
    ]
    # Numbers, not text, so the table sorts by value in the browser.
    assert pd.api.types.is_float_dtype(df["Annualized return (%)"])


def test_assets_df_absent_before_v32():
    assert assets_df(v32_namespace(assets=None)) is None


def test_constraints_markdown():
    assert constraints_markdown(CAPPED) == (
        "**Constraints:** long-only (no short positions), fully invested (weights "
        "sum to 100%), at most 40.0% in any one ticker."
    )
    assert constraints_markdown(UNCAPPED).endswith("no binding per-ticker weight cap.")
    assert constraints_markdown(None) == ""


def test_assets_caption_depends_on_cap():
    capped = assets_caption_markdown(CAPPED)
    assert "With at most 40.0% allowed in any one ticker" in capped
    assert "not a feasible portfolio" in capped
    for constraints in (UNCAPPED, None):
        caption = assets_caption_markdown(constraints)
        assert "not a feasible portfolio" not in caption
        assert "for reference" in caption


def test_near_frontier_sentence_tiny_share():
    # 0.0004 is 0.04%, which would round to 0.0%.
    assert near_frontier_sentence(random_stats()) == (
        "Less than 0.1% of the 10,000 random portfolios lie within 1 percentage "
        "point of the efficient frontier. Uniform draws cluster around the "
        "equal-weight portfolio and rarely reach the corners of the constraint "
        "set, where the frontier's high-return portfolios are."
    )


def test_near_frontier_sentence_other_shares():
    assert near_frontier_sentence(random_stats(share=0.1234, tolerance=0.5)).startswith(
        "Only 12.3% of the 10,000 random portfolios lie within 0.5 percentage points"
    )
    assert near_frontier_sentence(random_stats(share=0.0)).startswith(
        "None of the 10,000 random portfolios"
    )
    assert near_frontier_sentence(None) == ""


def test_portfolio_caption_keeps_illustrative_caption():
    v32 = portfolio_caption_markdown(v32_namespace())
    assert v32.startswith(f"*{RANDOM_PORTFOLIOS_CAPTION} Less than 0.1%")
    assert v32.endswith("*")
    # Before v32, or without metadata: only the illustrative caption.
    pre_v32 = f"*{RANDOM_PORTFOLIOS_CAPTION}*"
    assert portfolio_caption_markdown(v32_namespace(random_portfolios=None)) == pre_v32
    assert portfolio_caption_markdown(None) == pre_v32


def test_random_portfolios_details_markdown():
    text = random_portfolios_details_markdown(v32_namespace())
    assert "**Count:** 10,000 random portfolios" in text
    assert "32.20% among the random portfolios, vs. 50.00% for the maximum return" in text
    assert "11.30% for the random portfolios, vs. 11.20% for the equal-weight" in text
    assert "5.90 percentage points, vs. 6.40 expected" in text
    assert "annualized (arithmetic), in percent" in text
    assert "investment" not in text.lower()
    # Without the maximum return portfolio, no comparison is made.
    no_max = random_portfolios_details_markdown(v32_namespace(max_return=None))
    assert "maximum return portfolio" not in no_max
    assert random_portfolios_details_markdown(v32_namespace(random_portfolios=None)) == ""


def test_plot_marks_max_return_portfolio():
    metadata = SimpleNamespace(
        risk_free_rate=RISK_FREE_RATE, periods_per_year=252, max_return=MAX_RETURN
    )
    fig = portfolio_optimization_figure(plot_data(MAX_SHARPE_X), metadata)
    ax = fig.axes[0]
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels[-1] == "Maximum return portfolio"
    marker = [c for c in ax.collections if c.get_label() == "Maximum return portfolio"]
    assert len(marker) == 1
    assert marker[0].get_offsets().tolist() == [[35.0, 50.0]]
    # The axis limits include the marker, which lies beyond the other data.
    assert ax.get_xlim()[1] > 35.0 and ax.get_ylim()[1] > 50.0


def test_plot_without_max_return_portfolio():
    fig = portfolio_optimization_figure(plot_data(MAX_SHARPE_X), METADATA)
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert "Maximum return portfolio" not in labels

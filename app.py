import datetime
import os
import threading
from dataclasses import dataclass
import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError
from matplotlib.figure import Figure
from s3_downloader import S3Downloader
from data_loading import (
    ARIMA_SUFFIX,
    CONTRACT_VERSION,
    FORECASTS_KEY,
    FORECASTS_METADATA_KEY,
    NAIVE_SUFFIX,
    OPTIMIZATION_METADATA_KEY,
    ORDER_SUFFIXES,
    PLOT_DATA_KEY,
    DataLoadError,
    ForecastsMetadata,
    OptimizationMetadata,
    XYData,
    arima_orders,
    available_dates,
    clean_weights,
    folder_date_warnings,
    forecast_tickers,
    last_actual_date,
    live_forecast_date,
    load_forecasts,
    load_forecasts_metadata,
    load_optimization_metadata,
    load_plot_data,
    naive_column,
    optimization_files_agree,
)

INPUT_FOLDER = "input"
# One bucket (DigitalOcean Space) holds two top-level folders (see the contract).
BUCKET_NAME_ENV = "FSF_FRONT_END_BUCKET_NAME"
PORTFOLIO_FOLDER_ENV = "FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER"
TIME_SERIES_FOLDER_ENV = "FSF_FRONT_END_TIME_SERIES_FOLDER"

# The contract's disclaimer, verbatim. Shown exactly once, at the bottom of
# the page; do not repeat it in captions, tooltips, or metric descriptions.
DISCLAIMER_MD = (
    "**Not investment advice.** Every value shown is a backtest or model output "
    "computed from historical prices, and none of it predicts future results or "
    "recommends any investment."
)
FORECASTS_METADATA_UNPUBLISHED = (
    "Accuracy metrics are not available for this date. The backend publishes "
    "them from interface contract v21 on."
)

# Time series plot: one fixed color per series (Okabe-Ito, colorblind-safe).
ACTUAL_COLOR = "#0072B2"
ARIMA_COLOR = "#D55E00"
NAIVE_COLOR = "#999999"
# Height of the time series plot in pixels (about twice Gradio's default,
# which was too short once the angled date labels took their space).
TIMESERIES_PLOT_HEIGHT = 480
ACTUAL_TOOLTIP = "Actual adjusted close"
NAIVE_TOOLTIP = "Naive: the most recent observed close"


@dataclass(frozen=True)
class DateData:
    """
    The backend files from one date folder. Each file is either loaded or
    replaced by an error message. All files come from the same folder, so
    dates are never mixed.

    all_forecasts_metadata.yml is optional (contract v21+):
    forecasts_metadata_published says whether the folder listing showed it.
    """

    date: str
    metadata: OptimizationMetadata | None
    metadata_error: str | None
    plot_data: dict[str, XYData] | None
    plot_data_error: str | None
    forecasts: pd.DataFrame | None
    forecasts_error: str | None
    forecasts_metadata: ForecastsMetadata | None
    forecasts_metadata_error: str | None
    forecasts_metadata_published: bool

    def complete(self):
        """
        Return True if every file the date folder has loaded without
        errors. A date without the optional forecasts metadata can still be
        complete.
        """
        return (
            self.metadata is not None
            and self.plot_data is not None
            and self.forecasts is not None
            and (self.forecasts_metadata is not None or not self.forecasts_metadata_published)
        )


def arima_model_label(p, d, q):
    """
    Describe the ARIMA model behind one forecast, for the plot tooltip.

    Uses the contract's suggested wording: orders on log prices, with the
    equivalent model on daily log returns.

    Parameters
    ----------
    p, d, q : int | pandas.NA
        The forecast's model orders; missing when not published.

    Returns
    -------
    str
        The model description.
    """
    if pd.isna(p) or pd.isna(q):
        return "ARIMA forecast (model orders not published for this date)"
    if pd.isna(d):
        return f"ARIMA({p}, 0, {q}) on daily log returns"
    return (
        f"ARIMA({p}, {d}, {q}) on log prices "
        f"(fit as ARIMA({p}, 0, {q}) on daily log returns)"
    )


def arima_short_label(p, d, q):
    """
    Short name of the ARIMA model behind one forecast, e.g. "ARIMA(1, 1, 0)".

    Parameters
    ----------
    p, d, q : int | pandas.NA
        The forecast's model orders; missing when not published.

    Returns
    -------
    str
        "ARIMA(p, d, q)" on log prices, "ARIMA(p, 0, q) on log returns" if
        `d` is not published, or "ARIMA" if no orders are published.
    """
    if pd.isna(p) or pd.isna(q):
        return "ARIMA"
    if pd.isna(d):
        return f"ARIMA({p}, 0, {q}) on log returns"
    return f"ARIMA({p}, {d}, {q})"


def format_value(value, spec, suffix=""):
    """
    Format a metric for display, showing "n/a" for missing values.

    Parameters
    ----------
    value : float | int | None
        The value.

    spec : str
        Format specification, e.g. ".2f".

    suffix : str
        Text appended after the number, e.g. "%".

    Returns
    -------
    str
        The formatted value.
    """
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def folder_name(env_var):
    """
    Read a top-level folder name from an environment variable.

    Parameters
    ----------
    env_var : str
        Name of the environment variable.

    Returns
    -------
    str | None
        The folder name without leading or trailing slashes, or None if
        the variable is not set or empty.
    """
    return os.getenv(env_var, "").strip("/") or None


def date_label(date):
    """
    Format a date for the date dropdown, e.g. "2026-10-09 (Fri)".

    Parameters
    ----------
    date : str
        Date in `YYYY-MM-DD` format.

    Returns
    -------
    str
        The date followed by its abbreviated weekday.
    """
    weekday = datetime.date.fromisoformat(date).strftime("%a")
    return f"{date} ({weekday})"


class FSFinance:
    def __init__(self):
        """
        Instantiate the app class and the S3 client. Data is not loaded
        here: the available dates are listed on every page load, and the
        files for a date are downloaded when that date is selected.

        Loaded dates are cached in memory, shared by all visitors. Nothing
        that depends on a visitor's selection is stored on this object.
        """
        load_dotenv()
        if not os.path.exists(INPUT_FOLDER):
            os.mkdir(INPUT_FOLDER)
            print(f"Folder '{INPUT_FOLDER}' created")
        else:
            print(f"Folder '{INPUT_FOLDER}' already exists")
        try:
            self.downloader = S3Downloader()
        except (BotoCoreError, ValueError) as error:
            print(f"Could not create the S3 client: {type(error).__name__}")
            self.downloader = None
        # Date -> (marker ETags, DateData). See date_data().
        self._cache = {}
        self._cache_lock = threading.Lock()

    def list_dates(self):
        """
        List the dates that are complete in both top-level folders, with
        the ETags of their marker files (`optimization_metadata.yml` and
        `all_forecasts.csv`) and of the optional `all_forecasts_metadata.yml`.
        Keys whose segment after the folder is not a valid date are ignored.

        Returns
        -------
        tuple[dict[str, tuple[str, str, str | None]], str | None]
            Available date -> (metadata ETag, forecasts ETag, forecasts
            metadata ETag or None if the date has no such file), newest
            first, and an error message (None if dates were found).
        """
        if self.downloader is None:
            return {}, "The connection to the data storage could not be set up."
        missing = [
            env_var
            for env_var in (BUCKET_NAME_ENV, PORTFOLIO_FOLDER_ENV, TIME_SERIES_FOLDER_ENV)
            if not folder_name(env_var)
        ]
        if missing:
            print(f"Missing configuration: {', '.join(missing)}")
            return {}, "The data storage location is not configured."
        bucket_name = os.getenv(BUCKET_NAME_ENV)
        portfolio_keys = self.downloader.list_keys(
            bucket_name, folder_name(PORTFOLIO_FOLDER_ENV)
        )
        forecast_keys = self.downloader.list_keys(
            bucket_name, folder_name(TIME_SERIES_FOLDER_ENV)
        )
        if portfolio_keys is None or forecast_keys is None:
            return {}, "The list of available dates could not be read from the data storage."
        markers = {
            date: (
                portfolio_keys[f"{date}/{OPTIMIZATION_METADATA_KEY}"],
                forecast_keys[f"{date}/{FORECASTS_KEY}"],
                forecast_keys.get(f"{date}/{FORECASTS_METADATA_KEY}"),
            )
            for date in available_dates(portfolio_keys, forecast_keys)
        }
        if not markers:
            return {}, "No dates with complete data are available yet."
        return markers, None

    def fetch_and_load(self, folder_env_var, date, filename, loader):
        """
        Download one file from `<folder>/<date>/` in the bucket into
        `input/<date>/` and load it.

        Parameters
        ----------
        folder_env_var : str
            Name of the environment variable holding the top-level folder
            name.

        date : str
            Date folder (`YYYY-MM-DD`).

        filename : str
            File name within the date folder.

        loader : Callable[[str], Any]
            Function from data_loading that loads and validates the file.

        Returns
        -------
        tuple[Any | None, str | None]
            (loaded data, None) on success, or (None, error message).
        """
        object_name = f"{folder_name(folder_env_var)}/{date}/{filename}"
        local_filename = os.path.join(INPUT_FOLDER, date, filename)
        if self.downloader is None or not self.downloader.download_file(
            os.getenv(BUCKET_NAME_ENV), object_name, local_filename
        ):
            return None, f"`{date}/{filename}` could not be downloaded."
        try:
            return loader(local_filename), None
        except DataLoadError as error:
            print(f"Could not load {object_name}: {error}")
            return None, str(error)

    def load_date(self, date, forecasts_metadata_published):
        """
        Download and load the files from one date folder. The optional
        all_forecasts_metadata.yml is downloaded only if the folder listing
        showed it.

        Parameters
        ----------
        date : str
            Date folder (`YYYY-MM-DD`).

        forecasts_metadata_published : bool
            True if the date folder has all_forecasts_metadata.yml.

        Returns
        -------
        DateData
            The loaded files or their error messages.
        """
        os.makedirs(os.path.join(INPUT_FOLDER, date), exist_ok=True)
        plot_data, plot_data_error = self.fetch_and_load(
            PORTFOLIO_FOLDER_ENV, date, PLOT_DATA_KEY, load_plot_data
        )
        metadata, metadata_error = self.fetch_and_load(
            PORTFOLIO_FOLDER_ENV,
            date,
            OPTIMIZATION_METADATA_KEY,
            load_optimization_metadata,
        )
        forecasts, forecasts_error = self.fetch_and_load(
            TIME_SERIES_FOLDER_ENV, date, FORECASTS_KEY, load_forecasts
        )
        if forecasts_metadata_published:
            forecasts_metadata, forecasts_metadata_error = self.fetch_and_load(
                TIME_SERIES_FOLDER_ENV,
                date,
                FORECASTS_METADATA_KEY,
                load_forecasts_metadata,
            )
        else:
            forecasts_metadata = None
            forecasts_metadata_error = FORECASTS_METADATA_UNPUBLISHED
        return DateData(
            date=date,
            metadata=metadata,
            metadata_error=metadata_error,
            plot_data=plot_data,
            plot_data_error=plot_data_error,
            forecasts=forecasts,
            forecasts_error=forecasts_error,
            forecasts_metadata=forecasts_metadata,
            forecasts_metadata_error=forecasts_metadata_error,
            forecasts_metadata_published=forecasts_metadata_published,
        )

    def date_data(self, date, marker_etags):
        """
        Return the data for a date, from the cache when possible.

        A cached date is reused only while the ETags of its marker files
        (and of the optional forecasts metadata) are unchanged, so a
        same-day backend re-run is downloaded again. Only complete dates are
        cached, so a failed download is retried on the next request. The
        lock makes concurrent requests for a date download it once and
        never read a half-written local file.

        Parameters
        ----------
        date : str
            Date folder (`YYYY-MM-DD`).

        marker_etags : tuple[str, str, str | None]
            ETags of the date's optimization metadata, forecasts, and
            forecasts metadata (None if the date has none).

        Returns
        -------
        DateData
            The loaded files or their error messages.
        """
        with self._cache_lock:
            cached = self._cache.get(date)
            if cached is not None and cached[0] == tuple(marker_etags):
                return cached[1]
            data = self.load_date(
                date, forecasts_metadata_published=marker_etags[2] is not None
            )
            if data.complete():
                self._cache[date] = (tuple(marker_etags), data)
            return data

    def selected_data(self, date, markers):
        """
        Return the data for a visitor's selected date. If nothing is
        selected, or the selection is not an available date, the newest
        available date is used.

        Parameters
        ----------
        date : str | None
            Value of the date dropdown.

        markers : dict[str, tuple[str, str, str | None]]
            Available dates and marker ETags from list_dates().

        Returns
        -------
        DateData | None
            The date's data, or None if no dates are available.
        """
        if not markers:
            return None
        if date not in markers:
            date = max(markers)
        return self.date_data(date, markers[date])

    def resolve_ticker(self, forecasts, ticker):
        """
        Return the ticker if the forecasts contain it, otherwise the first
        available ticker. Ticker sets can differ between dates.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        ticker : str | None
            Value of the ticker dropdown.

        Returns
        -------
        str
            A ticker present in the forecasts.
        """
        tickers = forecast_tickers(forecasts.columns)
        return ticker if ticker in tickers else tickers[0]

    def friendly_col_name(self, col_name):
        """
        Return a user-friendly version of a column name specifying
        whether the vlaue is actual or modeled, and if modeled,
        what kind of model it is.

        Parameters
        ----------
        col_name : str
            Original column name

        Returns
        -------
        str
            User-friendly column name.
        """
        if col_name.endswith(ARIMA_SUFFIX):
            return f"{col_name.removesuffix(ARIMA_SUFFIX)} (ARIMA forecast)"
        elif col_name.endswith(NAIVE_SUFFIX):
            return f"{col_name.removesuffix(NAIVE_SUFFIX)} (Naive forecast)"
        else:
            return f"{col_name} (Actual)"

    def long_df(self, forecasts, ticker):
        """
        Isolate actual, ARIMA-forecast, and naive-forecast adjusted close
        prices of a ticker and pivot them from the wide dataframe into a
        long format suitable for visualization with gr.LinePlot(). The
        naive series is included only when the date folder has it
        (contract v19+).

        Empty values are dropped: actual prices on the live forecast row,
        and ARIMA forecasts after a missing close (contract v20), where the
        line joins across the gap. A "Model" column describes each point
        for the tooltip, including each ARIMA forecast's own (p, d, q).
        Dates are "YYYY-MM-DD" strings with no time of day, because every
        price is a market close.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        ticker : str
            The ticker to be plotted.

        Returns
        -------
        pd.DataFrame
            DataFrame for plotting with gr.LinePlot(), with columns
            "Date", "Series", "Adjusted Close ($)", and "Model".
        """
        arima_col = f"{ticker}{ARIMA_SUFFIX}"
        naive_col = naive_column(forecasts, ticker)
        value_columns = [ticker, arima_col] + ([naive_col] if naive_col else [])
        orders = arima_orders(forecasts, ticker)
        arima_labels = pd.Series(
            [arima_model_label(p, d, q) for p, d, q in orders.itertuples(index=False)],
            index=orders.index,
        )
        select_df = forecasts[value_columns].reset_index()
        select_df = select_df.melt(
            id_vars="pred_date",
            value_vars=value_columns,
            var_name="Series",
            value_name="Adjusted Close ($)",
        )
        select_df = select_df.dropna(subset=["Adjusted Close ($)"])
        select_df["Model"] = np.where(
            select_df["Series"] == arima_col,
            select_df["pred_date"].map(arima_labels),
            np.where(select_df["Series"] == naive_col, NAIVE_TOOLTIP, ACTUAL_TOOLTIP),
        )
        select_df["Series"] = select_df["Series"].apply(self.friendly_col_name)
        select_df["Adjusted Close ($)"] = select_df["Adjusted Close ($)"].round(2)
        # Dates as "YYYY-MM-DD" text: every value is a daily close, and
        # Gradio 5.22 shows time-based x values in tooltips with a time of
        # day in the viewer's time zone, which would be misleading.
        select_df["pred_date"] = select_df["pred_date"].dt.strftime("%Y-%m-%d")
        select_df.rename(columns={"pred_date": "Date"}, inplace=True)
        return select_df

    def timeseries_plot(self, forecasts, ticker):
        """
        Create a gr.LinePlot() with actual prices and ARIMA and naive
        forecasts for a ticker. Each series has a fixed color, and hovering
        a point shows the model behind it. The app calls this inside a
        gr.render() function so the plot is rebuilt from scratch whenever
        the selected date or ticker changes.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        ticker : str
            The ticker to be plotted.

        Returns
        -------
        gr.LinePlot
            An interactive LinePlot with ticker information.
        """
        long_df = self.long_df(forecasts, ticker)
        min_value = float(0.9 * long_df["Adjusted Close ($)"].min())
        max_value = float(1.1 * long_df["Adjusted Close ($)"].max())
        colors = {
            self.friendly_col_name(ticker): ACTUAL_COLOR,
            self.friendly_col_name(f"{ticker}{ARIMA_SUFFIX}"): ARIMA_COLOR,
            self.friendly_col_name(f"{ticker}{NAIVE_SUFFIX}"): NAIVE_COLOR,
        }
        # Only series present in the data, so the legend never lists a
        # series the date folder does not have (e.g. naive before v19).
        color_map = {
            series: color
            for series, color in colors.items()
            if series in set(long_df["Series"])
        }
        chart = gr.LinePlot(
            value=long_df,
            title=f"{ticker}: actual close vs. one-day-ahead forecasts",
            x="Date",
            x_title="Date (market close)",
            # Dates are text categories; ISO dates sort chronologically, and
            # each trading day gets equal spacing (no weekend gaps).
            sort="x",
            x_label_angle=-45,
            y="Adjusted Close ($)",
            color="Series",
            color_map=color_map,
            tooltip=["Series", "Model"],
            y_lim=[min_value, max_value],
            height=TIMESERIES_PLOT_HEIGHT,
        )
        return chart

    def portfolio_weights_df(self, metadata):
        """
        Create a Pandas DataFrame of portfolio tickers and weights of the
        maximum Sharpe ratio portfolio. Floating-point noise is cleaned
        (see data_loading.clean_weights) and weights are rounded to two
        decimals for display.

        Parameters
        ----------
        metadata : OptimizationMetadata
            Loaded optimization metadata.

        Returns
        -------
        pd.DataFrame
            DataFrame of tickers and weights in the portfolio.
        """
        df_rows = [
            {"Ticker": ticker, "Weight": f"{weight:.2f}%", "Numeric Weight": weight}
            for ticker, weight in clean_weights(metadata.weights).items()
        ]
        df = pd.DataFrame(df_rows)
        df.sort_values(
            by=["Numeric Weight", "Ticker"], ascending=[False, True], inplace=True
        )
        df.drop("Numeric Weight", axis=1, inplace=True)
        return df

    def optimization_metadata_markdown(self, metadata):
        """
        Generate markdown of optimization metadata: data dates, annualized
        performance of the maximum Sharpe ratio and minimum variance
        portfolios, and the assumptions behind them.

        Parameters
        ----------
        metadata : OptimizationMetadata
            Loaded optimization metadata.

        Returns
        -------
        str
            String of markdown
        """
        m = metadata
        lines = [
            f"**Data as of:** {m.date_to}",
            "## Maximum Sharpe ratio portfolio",
            f"- Annualized return: {m.optimum.annualized_return:.2f}%",
            f"- Annualized risk (σ): {m.optimum.risk:.2f}%",
            f"- Annualized Sharpe ratio: {m.sharpe_ratio:.2f}",
            "## Minimum variance portfolio",
            f"- Annualized return: {m.min_var.annualized_return:.2f}%",
            f"- Annualized risk (σ): {m.min_var.risk:.2f}%",
            "## Assumptions",
            f"- **Price history:** daily adjusted closes from {m.date_from} to {m.date_to}. "
            "Expected returns and covariances are estimated from this window, so the "
            "optimum is noisy and sensitive to the dates chosen.",
            "- **Returns:** simple (not log) daily returns, annualized arithmetically: "
            f"mean daily return × {m.periods_per_year} and daily σ × √{m.periods_per_year}. "
            "This is not the compounded growth an investor would realize.",
            f"- **Risk-free rate:** {m.risk_free_rate:.2f}% per year, the 3-month U.S. "
            f"Treasury bill rate (FRED series DTB3) observed {m.risk_free_rate_date}.",
            "- **Sharpe ratio:** (annualized return − risk-free rate) / annualized risk.",
            "- **Weights** are in percent and rounded, so they may not sum to exactly 100%.",
        ]
        return os.linesep.join(lines)

    def portfolio_consistency_markdown(self, data):
        """
        Generate a warning if the optimization YAML and HDF5 files appear to
        come from different backend runs.

        Parameters
        ----------
        data : DateData
            Data for the selected date.

        Returns
        -------
        str
            Warning markdown, or an empty string if the files agree or
            either file is unavailable.
        """
        if data.metadata is None or data.plot_data is None:
            return ""
        if optimization_files_agree(data.metadata, data.plot_data):
            return ""
        return (
            "**Warning:** the plot and the table below appear to come from "
            "different backend runs. The backend may have been uploading new data "
            "for this date; reload the page in a minute to download it again."
        )

    def date_warning_markdown(self, data):
        """
        Generate a warning if the files in the selected date folder do not
        describe that date.

        Parameters
        ----------
        data : DateData
            Data for the selected date.

        Returns
        -------
        str
            Warning markdown, or an empty string if everything agrees.
        """
        warnings = folder_date_warnings(
            data.date, data.metadata, data.forecasts, data.forecasts_metadata
        )
        if not warnings:
            return ""
        lines = [f"**Warning:** the files for {data.date} may describe another date."]
        lines += [f"- {warning}" for warning in warnings]
        return os.linesep.join(lines)

    def plot_portfolio_optimization(self, plot_data, metadata):
        """
        Plot a figure of the portfolio optimization. All values are
        annualized (arithmetic) and in percent. NaN points on the
        efficient frontier (infeasible target returns) break the line.
        The figure is created without pyplot, which is not thread-safe.

        Parameters
        ----------
        plot_data : dict[str, XYData]
            Loaded plot data.

        metadata : OptimizationMetadata | None
            Loaded optimization metadata, used to label the risk-free rate.

        Returns
        -------
        Figure
            A Matplotlib Figure object o be displayed on the UI.
        """
        tangency_label = "Capital market line"
        if metadata is not None:
            tangency_label += f" (risk-free rate {metadata.risk_free_rate:.2f}%)"
        fig = Figure()
        ax = fig.subplots(nrows=1, ncols=1)
        ax.plot(
            plot_data["efficient_frontier"].xs,
            plot_data["efficient_frontier"].ys,
            c="#009E73",
            zorder=1,
            label="Efficient frontier",
        )
        ax.plot(
            plot_data["tangency_line"].xs,
            plot_data["tangency_line"].ys,
            c="#F0E442",
            zorder=1,
            label=tangency_label,
        )
        ax.scatter(
            plot_data["simulated_portfolios"].xs,
            plot_data["simulated_portfolios"].ys,
            alpha=0.1,
            s=2,
            c="#0072B2",
            zorder=10,
            label="Random long-only portfolios",
        )
        ax.scatter(
            plot_data["max_sharpe_ratio"].xs,
            plot_data["max_sharpe_ratio"].ys,
            c="#CC79A7",
            marker="*",
            s=200,
            zorder=10,
            label="Max Sharpe ratio portfolio",
        )
        ax.scatter(
            plot_data["min_var_portfolio"].xs,
            plot_data["min_var_portfolio"].ys,
            c="#E69F00",
            zorder=10,
            label="Minimum variance portfolio",
        )
        ax.set_xlabel("Annualized Risk (σ, %)")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title("Portfolio Optimization (annualized, arithmetic)")
        ax.legend(fontsize="small")
        return fig

    def live_forecast_markdown(self, forecasts, ticker):
        """
        Generate markdown for a ticker's live (next trading day) forecast:
        the ARIMA forecast with its model orders, and the naive forecast.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        ticker : str
            The ticker.

        Returns
        -------
        str
            Markdown string for UI, or an empty string if the forecasts
            have no live row.
        """
        live_date = live_forecast_date(forecasts)
        if live_date is None:
            return ""
        lines = [f"### Live forecast for {live_date.date()}", "Adjusted close, US $:"]
        arima_value = forecasts.loc[live_date, f"{ticker}{ARIMA_SUFFIX}"]
        if pd.notna(arima_value):
            p, d, q = arima_orders(forecasts, ticker).loc[live_date]
            lines.append(f"- **{arima_short_label(p, d, q)}:** ${arima_value:.2f}")
        else:
            lines.append("- **ARIMA:** no forecast (the previous close is missing)")
        naive_col = naive_column(forecasts, ticker)
        if naive_col is not None and pd.notna(forecasts.loc[live_date, naive_col]):
            lines.append(f"- **Naive:** ${forecasts.loc[live_date, naive_col]:.2f}")
        return os.linesep.join(lines)

    def forecast_metrics_markdown(self, forecasts_metadata, unavailable_text, ticker):
        """
        Generate markdown of a ticker's forecast accuracy from
        all_forecasts_metadata.yml, in two parts for the two halves of the
        metrics panel: a table of RMSE, MAE, and MAPE for the ARIMA and
        naive forecasts and ARIMA directional accuracy, and a summary of
        relative MAE, skill, what they mean, and how to read the metrics
        (metrics_note_markdown). Nothing is computed here;
        the backend publishes every metric so all surfaces report the same
        numbers.

        Parameters
        ----------
        forecasts_metadata : ForecastsMetadata | None
            Loaded forecasts metadata, if available.

        unavailable_text : str | None
            Explanation shown when forecasts_metadata is None.

        ticker : str
            The ticker.

        Returns
        -------
        tuple[str, str]
            Markdown for the table half and for the summary half (empty
            when there are no metrics).
        """
        lines = ["### Forecast accuracy"]
        if forecasts_metadata is None:
            return os.linesep.join(lines + [unavailable_text or ""]), ""
        m = forecasts_metadata.metrics_for(ticker)
        if m is None:
            message = f"No accuracy metrics were published for {ticker}."
            return os.linesep.join(lines + [message]), ""
        if m.arima.directional_accuracy is None:
            direction = "n/a"
        else:
            direction = (
                f"{m.arima.directional_accuracy:.0%} "
                f"({format_value(m.arima.n_directional, 'd')} days)"
            )
        if m.skill is None:
            verdict = "Not enough scored days to compare ARIMA with the naive forecast."
        elif m.skill > 0:
            verdict = "ARIMA's average error was **smaller** than the naive forecast's."
        elif m.skill < 0:
            verdict = "ARIMA's average error was **larger** than the naive forecast's."
        else:
            verdict = "ARIMA's average error was the **same** as the naive forecast's."
        lines += [
            f"Scored on {format_value(m.n_evaluated, 'd')} of "
            f"{forecasts_metadata.target_trading_days} trading days "
            f"({m.evaluation_start_date or 'n/a'} to {m.evaluation_end_date or 'n/a'}).",
            "",
            "| | ARIMA | Naive |",
            "| --- | ---: | ---: |",
            f"| RMSE ($/share) | {format_value(m.arima.rmse, '.2f')} "
            f"| {format_value(m.naive.rmse, '.2f')} |",
            f"| MAE ($/share) | {format_value(m.arima.mae, '.2f')} "
            f"| {format_value(m.naive.mae, '.2f')} |",
            f"| MAPE (%) | {format_value(m.arima.mape, '.2f')} "
            f"| {format_value(m.naive.mape, '.2f')} |",
            f"| Directional accuracy | {direction} | — |",
        ]
        summary = [
            "### ARIMA vs. naive",
            f"**Relative MAE (ARIMA ÷ naive):** {format_value(m.relative_mae, '.3f')}  ",
            f"**Skill (1 − relative MAE):** {format_value(m.skill, '.3f')}",
            "",
            verdict,
            "",
            self.metrics_note_markdown(forecasts_metadata),
        ]
        return os.linesep.join(lines), os.linesep.join(summary)

    def metrics_note_markdown(self, forecasts_metadata):
        """
        Generate the contract's suggested wording for interpreting the
        forecast accuracy metrics, shown at the end of the summary half of
        the metrics panel.

        Parameters
        ----------
        forecasts_metadata : ForecastsMetadata | None
            Loaded forecasts metadata, if available.

        Returns
        -------
        str
            Markdown string for UI, or an empty string if no metrics are
            shown for this date.
        """
        if forecasts_metadata is None:
            return ""
        days = forecasts_metadata.target_trading_days
        return (
            "**Reading the accuracy metrics.** Accuracy over the last "
            f"{days} trading days, comparing each forecast with the actual close. "
            "Relative MAE below 1 (skill above 0) means ARIMA's errors were smaller "
            "than the naive forecast's. RMSE and MAE are in dollars per share; MAPE "
            "is in percent and can be compared across tickers. Directional accuracy "
            "is the share of days on which ARIMA predicted the direction of the move "
            "correctly; the naive forecast predicts no move, so it has none. "
            f"{days} days is a small sample, so treat these as indicative."
        )

    def timeseries_as_of_markdown(self, forecasts):
        """
        Generate the "Data as of" line shown directly below the Timeseries
        Analysis heading: the last date with an actual price.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        Returns
        -------
        str
            String of markdown
        """
        as_of = last_actual_date(forecasts)
        as_of_text = as_of.date() if as_of is not None else "no actual prices yet"
        return f"## Data as of: {as_of_text}"

    def timeseries_intro_markdown(self, forecasts, forecasts_metadata):
        """
        Generate the explanatory markdown of the Timeseries Analysis
        section, shown below the metrics panel inside a collapsed accordion
        (whose label serves as the heading), including the contract's suggested wording on
        how the forecasts are made, why d = 1, and the naive baseline.
        Parts about model orders and the naive forecast appear only when
        the date folder has those columns.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        forecasts_metadata : ForecastsMetadata | None
            Loaded forecasts metadata, if available, for the training
            window length.

        Returns
        -------
        str
            String of markdown
        """
        has_orders = any(
            str(column).endswith(ORDER_SUFFIXES) for column in forecasts.columns
        )
        has_naive = any(
            str(column).endswith(NAIVE_SUFFIX) for column in forecasts.columns
        )
        if forecasts_metadata is not None:
            window = f"the previous {forecasts_metadata.training_window_returns} trading days"
        else:
            window = "a short window of recent trading days"
        how = (
            f"For every day and ticker, an ARIMA model is refit on {window}, and its "
            "orders p and q are chosen again, so they can change from one day to the "
            "next. If a close is missing from the data, the next day's ARIMA "
            "forecast is skipped and the line joins across the gap."
        )
        if has_orders:
            how += " Hover over a forecast to see the model behind it."
        lines = [
            "Each point represents a one-day-ahead forecast of adjusted closing price (US $). Models are refit at "
            "each trading day using a rolling window of historical log returns. Training and model selection use "
            "only data available before the forecast date, preventing temporal leakage. Historical predictions are  "
            "evaluated against subsequently observed prices; the final point represents the next trading day's forecast. ",
            "### How these forecasts are made",
            how,
        ]
        if has_orders:
            lines += [
                "### Why d = 1",
                "The model is fit to daily log returns as ARIMA(p, 0, q). Turning "
                "prices into returns is one round of differencing, so the same model "
                "on log prices is ARIMA(p, 1, q). The tooltip shows the log-price "
                "version.",
            ]
        if has_naive:
            lines += [
                "### Naive forecast",
                "Today's close = yesterday's close (the most recent observed close). "
                "A model adds value only if its errors are smaller than the naive "
                "baseline's.",
            ]
        return os.linesep.join(lines)

    def run(self):
        """
        Run the Gradio app.

        The page is a single column: the date dropdown, then the Timeseries
        Analysis section, then the Portfolio Optimization section, each under
        a `#` heading. On every page load the available dates are listed and
        the newest is selected. Changing the date reloads both sections from
        that date's folder; changing the ticker redraws the time series plot. Plots
        are rebuilt with gr.render() rather than updated in place, because
        updating the value of an existing gr.LinePlot left the chart empty.
        """
        # Make the accuracy table fill its half of the metrics panel,
        # highlight the live forecast with a light yellow background (a
        # muted dark yellow in dark mode, where text is light) that stretches
        # to the bottom of the plot, and make the methods accordion's label
        # bold (its size stays the default).
        css = """
        .metrics-table table { width: 100% !important; }
        .live-forecast {
            background: #FFF7D1 !important;
            padding: 12px 16px !important;
            border-radius: 8px;
        }
        .dark .live-forecast { background: #4A4220 !important; }
        /* Stretch the live forecast box to the bottom of the plot beside it. */
        .live-forecast { flex-grow: 1 !important; }
        /* Everything inside the box (heading, text, list) 50% larger. zoom
           scales font sizes and spacing together, and only the text's
           elements are targeted, so the box keeps its size. */
        .prose.live-forecast > * { zoom: 1.5; }
        .methods-accordion .label-wrap > span:first-child {
            font-weight: 700 !important;
        }
        """
        with gr.Blocks(css=css) as app:
            # Available dates and marker ETags for this visitor's page load.
            marker_state = gr.State({})
            date_dropdown = gr.Dropdown(
                choices=[],
                value=None,
                label="Data as of (NYSE trading day)",
                interactive=True,
            )
            listing_md = gr.Markdown("")
            date_warning_md = gr.Markdown("")

            # Both sections are stacked on one page: Timeseries Analysis on
            # top, Portfolio Optimization below. Inner headings start at ##
            # so these # section headings stay the most prominent.
            gr.Markdown("# Timeseries Analysis")
            timeseries_as_of_md = gr.Markdown("")
            # The ticker and live forecast beside the plot; then a full-width
            # gray panel with the accuracy table (left half) and the
            # ARIMA-vs-naive summary followed by how to read the metrics
            # (right half); then the full-width explanation of the forecasts
            # and a horizontal rule.
            # equal_height lets the left column match the plot's height, so the
            # live forecast box (flex-grow in the CSS) reaches its bottom.
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    ticker_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Ticker",
                        visible=False,
                    )
                    live_forecast_md = gr.Markdown(
                        "", elem_classes=["live-forecast"], visible=False
                    )

                with gr.Column(scale=2):

                    @gr.render(
                        inputs=[date_dropdown, marker_state, ticker_dropdown],
                        triggers=[date_dropdown.change, ticker_dropdown.change],
                    )
                    def render_timeseries_plot(date, markers, ticker):
                        data = self.selected_data(date, markers)
                        if data is None or data.forecasts is None:
                            return
                        ticker = self.resolve_ticker(data.forecasts, ticker)
                        self.timeseries_plot(data.forecasts, ticker)

            with gr.Row(variant="panel", equal_height=False):
                with gr.Column(scale=1):
                    metrics_md = gr.Markdown("", elem_classes=["metrics-table"])
                with gr.Column(scale=1):
                    metrics_summary_md = gr.Markdown("")

            # The methods text is collapsed by default so it does not push the
            # Portfolio Optimization section far down the page.
            with gr.Accordion(
                "Click to learn more about walk-forward ARIMA, naive "
                "forecasting, and out-of-sample evaluation...",
                open=False,
                visible=False,
                elem_classes=["methods-accordion"],
            ) as methods_accordion:
                timeseries_intro_md = gr.Markdown("")
            gr.Markdown("---")

            gr.Markdown("# Portfolio Optimization")

            @gr.render(
                inputs=[date_dropdown, marker_state],
                triggers=[date_dropdown.change],
            )
            def render_portfolio(date, markers):
                data = self.selected_data(date, markers)
                if data is None:
                    return
                consistency_md = self.portfolio_consistency_markdown(data)
                if consistency_md:
                    gr.Markdown(consistency_md)

                with gr.Row():
                    if data.plot_data is not None:
                        gr.Plot(
                            self.plot_portfolio_optimization(
                                data.plot_data, data.metadata
                            )
                        )
                    else:
                        gr.Markdown(
                            f"**Portfolio plot unavailable.** {data.plot_data_error}"
                        )

                with gr.Row():
                    if data.metadata is not None:
                        with gr.Column():
                            gr.DataFrame(value=self.portfolio_weights_df(data.metadata))
                        with gr.Column():
                            gr.Markdown(
                                self.optimization_metadata_markdown(data.metadata)
                            )
                    else:
                        gr.Markdown(
                            f"**Portfolio summary unavailable.** {data.metadata_error}"
                        )

            gr.Markdown(
                f"Data format: interface contract v{CONTRACT_VERSION}. "
                "Data is updated manually, so check the \"Data as of\" dates."
            )

            gr.Markdown(DISCLAIMER_MD)

            def on_load():
                markers, error = self.list_dates()
                choices = [(date_label(date), date) for date in markers]
                newest = next(iter(markers), None)
                message = f"**No data to show.** {error}" if error else ""
                return gr.update(choices=choices, value=newest), markers, message

            def on_date_change(date, markers, ticker):
                data = self.selected_data(date, markers)
                hidden_tickers = gr.update(choices=[], value=None, visible=False)
                hidden_methods = gr.update(visible=False)
                if data is None:
                    return "", hidden_tickers, "", "", hidden_methods
                warning_md = self.date_warning_markdown(data)
                if data.forecasts is None:
                    # Shown under the heading, not in the collapsed accordion.
                    return (
                        warning_md,
                        hidden_tickers,
                        f"**Forecasts unavailable.** {data.forecasts_error}",
                        "",
                        hidden_methods,
                    )
                tickers = forecast_tickers(data.forecasts.columns)
                return (
                    warning_md,
                    gr.update(
                        choices=tickers,
                        value=self.resolve_ticker(data.forecasts, ticker),
                        visible=True,
                    ),
                    self.timeseries_as_of_markdown(data.forecasts),
                    self.timeseries_intro_markdown(
                        data.forecasts, data.forecasts_metadata
                    ),
                    gr.update(visible=True),
                )

            def on_ticker_text(date, markers, ticker):
                data = self.selected_data(date, markers)
                if data is None or data.forecasts is None:
                    return gr.update(value="", visible=False), "", ""
                ticker = self.resolve_ticker(data.forecasts, ticker)
                table_md, summary_md = self.forecast_metrics_markdown(
                    data.forecasts_metadata, data.forecasts_metadata_error, ticker
                )
                live_md = self.live_forecast_markdown(data.forecasts, ticker)
                # Hidden when there is no live row, so no empty yellow box shows.
                return (
                    gr.update(value=live_md, visible=bool(live_md)),
                    table_md,
                    summary_md,
                )

            app.load(
                on_load,
                outputs=[date_dropdown, marker_state, listing_md],
            )
            date_dropdown.change(
                on_date_change,
                inputs=[date_dropdown, marker_state, ticker_dropdown],
                outputs=[
                    date_warning_md,
                    ticker_dropdown,
                    timeseries_as_of_md,
                    timeseries_intro_md,
                    methods_accordion,
                ],
            )
            gr.on(
                triggers=[date_dropdown.change, ticker_dropdown.change],
                fn=on_ticker_text,
                inputs=[date_dropdown, marker_state, ticker_dropdown],
                outputs=[live_forecast_md, metrics_md, metrics_summary_md],
            )

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

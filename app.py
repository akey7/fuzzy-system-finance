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
    OPTIMIZATION_METADATA_KEY,
    PLOT_DATA_KEY,
    DataLoadError,
    OptimizationMetadata,
    XYData,
    available_dates,
    clean_weights,
    folder_date_warnings,
    forecast_tickers,
    last_actual_date,
    live_forecast_date,
    load_forecasts,
    load_optimization_metadata,
    load_plot_data,
    optimization_files_agree,
)

INPUT_FOLDER = "input"
MARKET_TIMEZONE = "America/New_York"
# One bucket (DigitalOcean Space) holds two top-level folders (see the contract).
BUCKET_NAME_ENV = "FSF_FRONT_END_BUCKET_NAME"
PORTFOLIO_FOLDER_ENV = "FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER"
TIME_SERIES_FOLDER_ENV = "FSF_FRONT_END_TIME_SERIES_FOLDER"

DISCLAIMER_MD = (
    "**Not investment advice.** This is a hobby data science project. Every "
    "value shown is a backtest or model output computed from historical prices, "
    "and none of it predicts future results or recommends any investment."
)


@dataclass(frozen=True)
class DateData:
    """
    The three backend files from one date folder. Each file is either
    loaded or replaced by an error message. All files come from the same
    folder, so dates are never mixed.
    """

    date: str
    metadata: OptimizationMetadata | None
    metadata_error: str | None
    plot_data: dict[str, XYData] | None
    plot_data_error: str | None
    forecasts: pd.DataFrame | None
    forecasts_error: str | None

    def complete(self):
        """
        Return True if all three files loaded without errors.
        """
        return (
            self.metadata is not None
            and self.plot_data is not None
            and self.forecasts is not None
        )


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
        `all_forecasts.csv`). Keys whose segment after the folder is not a
        valid date are ignored.

        Returns
        -------
        tuple[dict[str, tuple[str, str]], str | None]
            Available date -> (metadata ETag, forecasts ETag), newest
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

    def load_date(self, date):
        """
        Download and load all three files from one date folder.

        Parameters
        ----------
        date : str
            Date folder (`YYYY-MM-DD`).

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
        return DateData(
            date=date,
            metadata=metadata,
            metadata_error=metadata_error,
            plot_data=plot_data,
            plot_data_error=plot_data_error,
            forecasts=forecasts,
            forecasts_error=forecasts_error,
        )

    def date_data(self, date, marker_etags):
        """
        Return the data for a date, from the cache when possible.

        A cached date is reused only while the ETags of its marker files
        are unchanged, so a same-day backend re-run is downloaded again.
        Only complete dates are cached, so a failed download is retried on
        the next request. The lock makes concurrent requests for a date
        download it once and never read a half-written local file.

        Parameters
        ----------
        date : str
            Date folder (`YYYY-MM-DD`).

        marker_etags : tuple[str, str]
            ETags of the date's metadata and forecasts files.

        Returns
        -------
        DateData
            The loaded files or their error messages.
        """
        with self._cache_lock:
            cached = self._cache.get(date)
            if cached is not None and cached[0] == tuple(marker_etags):
                return cached[1]
            data = self.load_date(date)
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

        markers : dict[str, tuple[str, str]]
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
        else:
            return f"{col_name} (Actual)"

    def long_df(self, forecasts, ticker):
        """
        Isolate predicted and actual adjusted close prices of
        a ticker and pivots the wide dataframe with the predicted
        and actual them into a long format suitable for
        visualization with gr.LinePlot(). Empty actual prices (the live
        forecast row) are dropped. Dates are placed at 16:00 US Eastern
        (market close) so they display on the correct day.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        ticker : str
            The ticker to be plotted.

        Returns
        -------
        pd.DataFrame
            DataFrame for plotting with gr.LinePlot().
        """
        select_df = forecasts[[ticker, f"{ticker}{ARIMA_SUFFIX}"]].reset_index()
        select_df = select_df.melt(
            id_vars="pred_date",
            value_vars=[ticker, f"{ticker}{ARIMA_SUFFIX}"],
            var_name="Ticker",
            value_name="Adjusted Close ($)",
        )
        select_df = select_df.dropna(subset=["Adjusted Close ($)"])
        select_df["Ticker"] = select_df["Ticker"].apply(self.friendly_col_name)
        select_df["Adjusted Close ($)"] = select_df["Adjusted Close ($)"].round(2)
        # Add the offset in NumPy with an explicit unit; pandas 2.x with
        # NumPy >= 2.5 warns about a unitless timedelta for Series + Timedelta.
        market_close = select_df["pred_date"].to_numpy() + np.timedelta64(16, "h")
        select_df["pred_date"] = pd.Series(
            market_close, index=select_df.index
        ).dt.tz_localize(MARKET_TIMEZONE)
        select_df.rename(columns={"pred_date": "Date"}, inplace=True)
        return select_df

    def timeseries_plot(self, forecasts, ticker):
        """
        Create a gr.LinePlot() with actual and predicted prices for
        a ticker. The app calls this inside a gr.render() function so the
        plot is rebuilt from scratch whenever the selected date or ticker
        changes.

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
        chart = gr.LinePlot(
            value=long_df,
            title=f"{ticker}: actual vs. one-day-ahead ARIMA forecast",
            x="Date",
            y="Adjusted Close ($)",
            color="Ticker",
            y_lim=[min_value, max_value],
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
        warnings = folder_date_warnings(data.date, data.metadata, data.forecasts)
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

    def arima_backtest_df(self, forecasts, ticker):
        """
        Return the backtest rows for a ticker: rows that have both an
        actual price and an ARIMA forecast.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        ticker : str
            The ticker.

        Returns
        -------
        pd.DataFrame
            Actual and forecast columns for the ticker on backtest rows.
        """
        columns = [ticker, f"{ticker}{ARIMA_SUFFIX}"]
        return forecasts[columns].dropna()

    def calc_rmse(self, forecasts, ticker):
        """
        Calculate the RMSE of the walk-forward ARIMA model of an asset
        over its backtest rows, in US dollars.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        ticker : str
            The ticker.

        Returns
        -------
        float
            RMSE of ARIMA model, or NaN if there are no backtest rows.
        """
        backtest = self.arima_backtest_df(forecasts, ticker)
        if backtest.empty:
            return float("nan")
        errors = backtest[ticker] - backtest[f"{ticker}{ARIMA_SUFFIX}"]
        return float(np.sqrt(np.mean(np.square(errors))))

    def arima_rmse_message(self, forecasts, ticker):
        """
        Turn the raw RMSE value and the live forecast into a markdown
        string for display on the UI.

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts from data_loading.load_forecasts().

        ticker : str
            The ticker.

        Returns
        -------
        str
            Markdown string for UI.
        """
        backtest = self.arima_backtest_df(forecasts, ticker)
        if backtest.empty:
            lines = ["### Backtest RMSE", "No backtest rows with actual prices yet."]
        else:
            first = backtest.index.min().date()
            last = backtest.index.max().date()
            lines = [
                "### Backtest RMSE",
                f"# ${self.calc_rmse(forecasts, ticker):.2f}",
                f"Root mean squared error of the one-day-ahead forecast against the "
                f"actual adjusted close, in US dollars, over {len(backtest)} trading "
                f"days ({first} to {last}).",
            ]
        live_date = live_forecast_date(forecasts)
        if live_date is not None:
            live_value = forecasts.loc[live_date, f"{ticker}{ARIMA_SUFFIX}"]
            if pd.notna(live_value):
                lines += [
                    "### Live forecast",
                    f"**{live_date.date()}:** ${live_value:.2f} adjusted close",
                ]
        return os.linesep.join(lines)

    def timeseries_intro_markdown(self, forecasts):
        """
        Generate the explanatory markdown at the top of the time series tab.

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
        lines = [
            "## One-day-ahead ARIMA forecasts",
            "Each ARIMA point forecasts that day's adjusted close (US $) using only "
            "earlier prices, from a model refit every day on a short sliding window "
            "of recent prices. Days with an actual price form the backtest; the "
            "final day, which has no actual price yet, is the live forecast.",
            f"**Data as of:** {as_of_text}",
            "### Select a ticker to view the model:",
        ]
        return os.linesep.join(lines)

    def run(self):
        """
        Run the Gradio app.

        On every page load the available dates are listed and the newest
        is selected. Changing the date reloads both tabs from that date's
        folder; changing the ticker redraws the time series plot. Plots
        are rebuilt with gr.render() rather than updated in place, because
        updating the value of an existing gr.LinePlot left the chart empty.
        """
        with gr.Blocks() as app:
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

            with gr.Tabs():
                with gr.Tab("Portfolio Optimization"):

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
                                    gr.DataFrame(
                                        value=self.portfolio_weights_df(data.metadata)
                                    )
                                with gr.Column():
                                    gr.Markdown(
                                        self.optimization_metadata_markdown(
                                            data.metadata
                                        )
                                    )
                            else:
                                gr.Markdown(
                                    f"**Portfolio summary unavailable.** {data.metadata_error}"
                                )

                with gr.Tab("Timeseries Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            timeseries_intro_md = gr.Markdown("")
                            ticker_dropdown = gr.Dropdown(
                                choices=[],
                                value=None,
                                label=None,
                                show_label=False,
                                visible=False,
                            )
                            arima_rmse_md = gr.Markdown("", container=True)

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
                if data is None:
                    return "", hidden_tickers, ""
                warning_md = self.date_warning_markdown(data)
                if data.forecasts is None:
                    return (
                        warning_md,
                        hidden_tickers,
                        f"**Forecasts unavailable.** {data.forecasts_error}",
                    )
                tickers = forecast_tickers(data.forecasts.columns)
                return (
                    warning_md,
                    gr.update(
                        choices=tickers,
                        value=self.resolve_ticker(data.forecasts, ticker),
                        visible=True,
                    ),
                    self.timeseries_intro_markdown(data.forecasts),
                )

            def on_ticker_text(date, markers, ticker):
                data = self.selected_data(date, markers)
                if data is None or data.forecasts is None:
                    return ""
                ticker = self.resolve_ticker(data.forecasts, ticker)
                return self.arima_rmse_message(data.forecasts, ticker)

            app.load(
                on_load,
                outputs=[date_dropdown, marker_state, listing_md],
            )
            date_dropdown.change(
                on_date_change,
                inputs=[date_dropdown, marker_state, ticker_dropdown],
                outputs=[date_warning_md, ticker_dropdown, timeseries_intro_md],
            )
            gr.on(
                triggers=[date_dropdown.change, ticker_dropdown.change],
                fn=on_ticker_text,
                inputs=[date_dropdown, marker_state, ticker_dropdown],
                outputs=[arima_rmse_md],
            )

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

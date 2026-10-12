import os
import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError
import matplotlib.pyplot as plt
from s3_downloader import S3Downloader
from data_loading import (
    ARIMA_SUFFIX,
    CONTRACT_VERSION,
    FORECASTS_KEY,
    OPTIMIZATION_METADATA_KEY,
    PLOT_DATA_KEY,
    DataLoadError,
    clean_weights,
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

DISCLAIMER_MD = (
    "**Not investment advice.** This is a hobby data science project. Every "
    "value shown is a backtest or model output computed from historical prices, "
    "and none of it predicts future results or recommends any investment."
)


class FSFinance:
    def __init__(self):
        """
        Instantiate the app class, download the backend files described in
        INTERFACE_CONTRACT.md, and load them. A file that cannot be
        downloaded or does not satisfy the contract is stored as None
        alongside an error message, so the app still starts and shows the
        message instead of crashing.
        """
        load_dotenv()
        if not os.path.exists(INPUT_FOLDER):
            os.mkdir(INPUT_FOLDER)
            print(f"Folder '{INPUT_FOLDER}' created")
        else:
            print(f"Folder '{INPUT_FOLDER}' already exists")
        try:
            s3d = S3Downloader()
        except (BotoCoreError, ValueError) as error:
            print(f"Could not create the S3 client: {type(error).__name__}")
            s3d = None
        self.plot_data, self.plot_data_error = self.fetch_and_load(
            s3d, "PORTFOLIO_OPTIMIZATION_SPACE_NAME", PLOT_DATA_KEY, load_plot_data
        )
        self.metadata, self.metadata_error = self.fetch_and_load(
            s3d,
            "PORTFOLIO_OPTIMIZATION_SPACE_NAME",
            OPTIMIZATION_METADATA_KEY,
            load_optimization_metadata,
        )
        self.forecasts, self.forecasts_error = self.fetch_and_load(
            s3d, "TIME_SERIES_SPACE_NAME", FORECASTS_KEY, load_forecasts
        )

    def fetch_and_load(self, downloader, bucket_env_var, object_name, loader):
        """
        Download one bucket file into the input folder and load it.

        Parameters
        ----------
        downloader : S3Downloader | None
            Downloader to use. None means the S3 client could not be created.

        bucket_env_var : str
            Name of the environment variable holding the bucket name.

        object_name : str
            Bucket key of the file.

        loader : Callable[[str], Any]
            Function from data_loading that loads and validates the file.

        Returns
        -------
        tuple[Any | None, str | None]
            (loaded data, None) on success, or (None, error message).
        """
        local_filename = os.path.join(INPUT_FOLDER, object_name)
        if downloader is None or not downloader.download_file(
            os.getenv(bucket_env_var), object_name, local_filename
        ):
            return None, f"`{object_name}` could not be downloaded from the bucket."
        try:
            return loader(local_filename), None
        except DataLoadError as error:
            print(f"Could not load {object_name}: {error}")
            return None, str(error)

    def tickers(self):
        """
        Return the sorted tickers available for visualization from the
        forecasts DataFrame: those with both an actual-price column and
        an ARIMA forecast column.
        """
        return forecast_tickers(self.forecasts.columns)

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

    def long_df(self, ticker):
        """
        Isolate predicted and actual adjusted close prices of
        a ticker and pivots the wide dataframe with the predicted
        and actual them into a long format suitable for
        visualization with gr.LinePlot(). Empty actual prices (the live
        forecast row) are dropped. Dates are placed at 16:00 US Eastern
        (market close) so they display on the correct day.

        Parameters
        ----------
        ticker : str
            The ticker to be plotted.

        Returns
        -------
        pd.DataFrame
            DataFrame for plotting with gr.LinePlot().
        """
        select_df = self.forecasts[[ticker, f"{ticker}{ARIMA_SUFFIX}"]].reset_index()
        select_df = select_df.melt(
            id_vars="pred_date",
            value_vars=[ticker, f"{ticker}{ARIMA_SUFFIX}"],
            var_name="Ticker",
            value_name="Adjusted Close ($)",
        )
        select_df = select_df.dropna(subset=["Adjusted Close ($)"])
        select_df["Ticker"] = select_df["Ticker"].apply(self.friendly_col_name)
        select_df["Adjusted Close ($)"] = select_df["Adjusted Close ($)"].round(2)
        select_df["pred_date"] = (
            select_df["pred_date"] + pd.Timedelta(hours=16)
        ).dt.tz_localize(MARKET_TIMEZONE)
        select_df.rename(columns={"pred_date": "Date"}, inplace=True)
        return select_df

    def timeseries_plot(self, ticker):
        """
        Create a gr.LinePlot() with actual and predicted prices for
        a ticker. The app calls this inside a gr.render() function so the
        plot is rebuilt from scratch whenever the selected ticker changes.

        Parameters
        ----------
        ticker : str
            The ticker to be plotted.

        Returns
        -------
        gr.LinePlot
            An interactive LinePlot with ticker information.
        """
        long_df = self.long_df(ticker)
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

    def portfolio_weights_df(self):
        """
        Create a Pandas DataFrame of portfolio tickers and weights of the
        maximum Sharpe ratio portfolio. Floating-point noise is cleaned
        (see data_loading.clean_weights) and weights are rounded to two
        decimals for display.

        Returns
        -------
        pd.DataFrame
            DataFrame of tickers and weights in the portfolio.
        """
        df_rows = [
            {"Ticker": ticker, "Weight": f"{weight:.2f}%", "Numeric Weight": weight}
            for ticker, weight in clean_weights(self.metadata.weights).items()
        ]
        df = pd.DataFrame(df_rows)
        df.sort_values(
            by=["Numeric Weight", "Ticker"], ascending=[False, True], inplace=True
        )
        df.drop("Numeric Weight", axis=1, inplace=True)
        return df

    def optimization_metadata_markdown(self):
        """
        Generate markdown of optimization metadata: data dates, annualized
        performance of the maximum Sharpe ratio and minimum variance
        portfolios, and the assumptions behind them.

        Returns
        -------
        str
            String of markdown
        """
        m = self.metadata
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

    def portfolio_consistency_markdown(self):
        """
        Generate a warning if the optimization YAML and HDF5 files appear to
        come from different backend runs.

        Returns
        -------
        str
            Warning markdown, or an empty string if the files agree or
            either file is unavailable.
        """
        if self.metadata is None or self.plot_data is None:
            return ""
        if optimization_files_agree(self.metadata, self.plot_data):
            return ""
        return (
            "**Warning:** the plot and the table below appear to come from "
            "different backend runs. The backend may have been uploading new data "
            "when this app started; restart the app to download it again."
        )

    def plot_portfolio_optimization(self):
        """
        Plot a figure of the portfolio optimization. All values are
        annualized (arithmetic) and in percent. NaN points on the
        efficient frontier (infeasible target returns) break the line.

        Returns
        -------
        Figure
            A Matplotlib Figure object o be displayed on the UI.
        """
        plot_data = self.plot_data
        tangency_label = "Capital market line"
        if self.metadata is not None:
            tangency_label += f" (risk-free rate {self.metadata.risk_free_rate:.2f}%)"
        fig, ax = plt.subplots(nrows=1, ncols=1)
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

    def arima_backtest_df(self, ticker):
        """
        Return the backtest rows for a ticker: rows that have both an
        actual price and an ARIMA forecast.

        Parameters
        ----------
        ticker : str
            The ticker.

        Returns
        -------
        pd.DataFrame
            Actual and forecast columns for the ticker on backtest rows.
        """
        columns = [ticker, f"{ticker}{ARIMA_SUFFIX}"]
        return self.forecasts[columns].dropna()

    def calc_rmse(self, ticker):
        """
        Calculate the RMSE of the walk-forward ARIMA model of an asset
        over its backtest rows, in US dollars.

        Parameters
        ----------
        ticker : str
            The ticker.

        Returns
        -------
        float
            RMSE of ARIMA model, or NaN if there are no backtest rows.
        """
        backtest = self.arima_backtest_df(ticker)
        if backtest.empty:
            return float("nan")
        errors = backtest[ticker] - backtest[f"{ticker}{ARIMA_SUFFIX}"]
        return float(np.sqrt(np.mean(np.square(errors))))

    def arima_rmse_message(self, ticker):
        """
        Turn the raw RMSE value and the live forecast into a markdown
        string for display on the UI.

        Parameters
        ----------
        ticker : str
            The ticker.

        Returns
        -------
        str
            Markdown string for UI.
        """
        backtest = self.arima_backtest_df(ticker)
        if backtest.empty:
            lines = ["### Backtest RMSE", "No backtest rows with actual prices yet."]
        else:
            first = backtest.index.min().date()
            last = backtest.index.max().date()
            lines = [
                "### Backtest RMSE",
                f"# ${self.calc_rmse(ticker):.2f}",
                f"Root mean squared error of the one-day-ahead forecast against the "
                f"actual adjusted close, in US dollars, over {len(backtest)} trading "
                f"days ({first} to {last}).",
            ]
        live_date = live_forecast_date(self.forecasts)
        if live_date is not None:
            live_value = self.forecasts.loc[live_date, f"{ticker}{ARIMA_SUFFIX}"]
            if pd.notna(live_value):
                lines += [
                    "### Live forecast",
                    f"**{live_date.date()}:** ${live_value:.2f} adjusted close",
                ]
        return os.linesep.join(lines)

    def timeseries_intro_markdown(self):
        """
        Generate the explanatory markdown at the top of the time series tab.

        Returns
        -------
        str
            String of markdown
        """
        as_of = last_actual_date(self.forecasts)
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
        """
        with gr.Blocks() as app:
            gr.Markdown(DISCLAIMER_MD)

            with gr.Tabs():
                with gr.Tab("Portfolio Optimization"):
                    consistency_md = self.portfolio_consistency_markdown()
                    if consistency_md:
                        gr.Markdown(consistency_md)

                    with gr.Row():
                        if self.plot_data is not None:
                            gr.Plot(self.plot_portfolio_optimization())
                        else:
                            gr.Markdown(
                                f"**Portfolio plot unavailable.** {self.plot_data_error}"
                            )

                    with gr.Row():
                        if self.metadata is not None:
                            with gr.Column():
                                gr.DataFrame(value=self.portfolio_weights_df())
                            with gr.Column():
                                gr.Markdown(self.optimization_metadata_markdown())
                        else:
                            gr.Markdown(
                                f"**Portfolio summary unavailable.** {self.metadata_error}"
                            )

                with gr.Tab("Timeseries Analysis"):
                    if self.forecasts is None:
                        gr.Markdown(
                            f"**Forecasts unavailable.** {self.forecasts_error}"
                        )
                    else:
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown(self.timeseries_intro_markdown())
                                ticker_dropdown = gr.Dropdown(
                                    choices=self.tickers(),
                                    label=None,
                                    show_label=False,
                                    value=self.tickers()[0],
                                )
                                arima_rmse_md = gr.Markdown(
                                    self.arima_rmse_message(self.tickers()[0]),
                                    container=True,
                                )

                            with gr.Column(scale=2):
                                # Rebuild the plot on each ticker change rather
                                # than updating the value of an existing
                                # gr.LinePlot, which left the chart empty.
                                @gr.render(inputs=[ticker_dropdown])
                                def render_timeseries_plot(choice):
                                    if choice:
                                        self.timeseries_plot(choice)

                        def ticker_change(choice):
                            return self.arima_rmse_message(choice)

                        ticker_dropdown.change(
                            ticker_change,
                            inputs=[ticker_dropdown],
                            outputs=[arima_rmse_md],
                        )

            gr.Markdown(
                f"Data format: interface contract v{CONTRACT_VERSION}. "
                "Data is updated manually, so check the \"Data as of\" dates."
            )

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

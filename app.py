import os
import pytz
import gradio as gr
import pandas as pd
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error
import boto3
from botocore.client import Config
import h5py
import matplotlib.pyplot as plt


class FSFinance:
    def __init__(self):
        """
        Instantiate the app class and read the csv that contains
        the data to be visualized.
        """
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(hf_token, add_to_git_credential=True)
        hf_datset = os.getenv("HF_DATASET")
        dataset = load_dataset(hf_datset)
        self.df = dataset["train"].to_pandas()
        self.df["pred_date"] = pd.to_datetime(self.df["pred_date"])
        region_name = os.getenv("FSF_FRONT_END_BUCKET_REGION")
        endpoint_url = os.getenv("FSF_FRONT_END_BUCKET_ENDPOINT")
        aws_access_key_id = os.getenv("FSF_FRONT_END_BUCKET_READ_ONLY_KEY_ID")
        aws_secret_access_key = os.getenv("FSF_FRONT_END_BUCKET_READ_ONLY")
        bucket_name = "portfolio-optimization"
        object_name = "portfolio_optimization_plot_data.h5"
        self.portfolio_optimization_plot_data_path = os.path.join("input", object_name)
        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=region_name,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )
        client.download_file(
            bucket_name, object_name, self.portfolio_optimization_plot_data_path
        )
        print(f"Downloaded {self.portfolio_optimization_plot_data_path}")

    def tickers(self):
        """
        Return the sorted tickers available for visualization from the
        underlying DataFrame.
        """
        return sorted([ticker for ticker in self.df.columns if "_" not in ticker])

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
        if "_arima" in col_name:
            return f"{col_name.replace('_arima', '')} (ARIMA model)"
        elif "_hw" in col_name:
            return f"{col_name.replace('_hw', '')} (Holt-Winters model)"
        else:
            return f"{col_name} (Actual)"

    def long_df(self, ticker):
        """
        Isolate predicted and actual adjusted close prices of
        a ticker and pivots the wide dataframe with the predicted
        and actual them into a long format suitable for
        visualization with gr.LinePlot().

        Parameters
        ----------
        ticker : str
            The ticker to be plotted.

        Returns
        -------
        pd.DataFrame
            DataFrame for plotting with gr.LinePlot().
        """
        select_df = self.df[
            ["pred_date", ticker, f"{ticker}_arima", f"{ticker}_hw"]
        ].copy()
        select_df = select_df.melt(
            id_vars="pred_date",
            value_vars=[ticker, f"{ticker}_arima", f"{ticker}_hw"],
            var_name="Ticker",
            value_name="Adjusted Close ($)",
        )
        select_df["Ticker"] = select_df["Ticker"].apply(self.friendly_col_name)
        select_df["Adjusted Close ($)"] = select_df["Adjusted Close ($)"].apply(
            lambda x: round(x, 2)
        )
        eastern = pytz.timezone("US/Eastern")
        select_df["pred_date"] = select_df["pred_date"].apply(
            lambda x: pd.Timestamp(x.replace(hour=16, minute=0, second=0), tz=eastern)
        )
        select_df.rename(columns={"pred_date": "Date"}, inplace=True)
        return select_df

    def timeseries_plot(self, ticker):
        """
        Create a gr.LinePlot() with actual and predicted prices for
        a ticker.

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
        min_value = 0.9 * min(long_df.loc[:, "Adjusted Close ($)"])
        max_value = 1.1 * max(long_df.loc[:, "Adjusted Close ($)"])
        chart = gr.LinePlot(
            value=long_df,
            title=ticker,
            x="Date",
            y="Adjusted Close ($)",
            color="Ticker",
            y_lim=[min_value, max_value],
        )
        return chart

    def plot_portfolio_optimization(self):
        with h5py.File(self.portfolio_optimization_plot_data_path, "r") as hf:
            efficient_frontier_xs = hf["efficient_frontier/xs"][:]
            efficient_frontier_ys = hf["efficient_frontier/ys"][:]
            tangency_line_xs = hf["tangency_line/xs"][:]
            tangency_line_ys = hf["tangency_line/ys"][:]
            simulated_portfolios_xs = hf["simulated_portfolios/xs"][:]
            simulated_portfolios_ys = hf["simulated_portfolios/ys"][:]
            max_sharpe_ratio_xs = hf["max_sharpe_ratio/xs"][:]
            max_sharpe_ratio_ys = hf["max_sharpe_ratio/ys"][:]
            min_var_portfolio_xs = hf["min_var_portfolio/xs"][:]
            min_var_portfolio_ys = hf["min_var_portfolio/ys"][:]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(
            efficient_frontier_xs,
            efficient_frontier_ys,
            c="#009E73",
            zorder=1,
            label="Efficient Frontier",
        )
        ax.plot(
            tangency_line_xs,
            tangency_line_ys,
            c="#F0E442",
            zorder=1,
            label="Tangency line",
        )
        ax.scatter(
            simulated_portfolios_xs,
            simulated_portfolios_ys,
            alpha=0.1,
            s=2,
            c="#0072B2",
            zorder=10,
            label="Portfolios",
        )
        ax.scatter(
            max_sharpe_ratio_xs,
            max_sharpe_ratio_ys,
            c="#CC79A7",
            marker="*",
            s=200,
            zorder=10,
            label="Max Sharpe Ratio Portfolio",
        )
        ax.scatter(
            min_var_portfolio_xs,
            min_var_portfolio_ys,
            c="#E69F00",
            zorder=10,
            label="Min Var Portfolio",
        )
        ax.set_xlabel("Risk (σ)")
        ax.set_ylabel("Daily Returns (%)")
        ax.set_title("Portfolio Optimization")
        ax.legend()
        return fig

    def calc_rmse(self, ticker):
        y_actual = self.df[ticker][:-1]
        y_pred_arima = self.df[f"{ticker}_arima"][:-1]
        y_pred_hw = self.df[f"{ticker}_hw"][:-1]
        rmse_arima = root_mean_squared_error(y_actual, y_pred_arima)
        rmse_hw = root_mean_squared_error(y_actual, y_pred_hw)
        return rmse_arima, rmse_hw

    def arima_rmse_message(self, ticker):
        rmse_arima, _ = self.calc_rmse(ticker)
        rmse_arima_fmt = f"{rmse_arima:.2f}"
        return f"### ARIMA RMSE:{os.linesep}# {rmse_arima_fmt}"

    def hw_rmse_message(self, ticker):
        _, rmse_hw = self.calc_rmse(ticker)
        rmse_hw_fmt = f"{rmse_hw:.2f}"
        return f"### Holt-Winters RMSE:{os.linesep}# {rmse_hw_fmt}"

    def run(self):
        """
        Run the Gradio app.
        """
        with gr.Blocks() as app:

            def ticker_change(choice):
                return (
                    self.timeseries_plot(choice),
                    self.arima_rmse_message(choice),
                    self.hw_rmse_message(choice),
                )

            with gr.Row():
                portfolio_optimization_plot = gr.Plot(
                    self.plot_portfolio_optimization()
                )

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("### Select a ticker or index:")
                    ticker_dropdown = gr.Dropdown(
                        choices=self.tickers(),
                        label=None,
                        show_label=False,
                        value=self.tickers()[0],
                    )
            with gr.Row():
                with gr.Column():
                    arima_rmse_md = gr.Markdown(
                        self.arima_rmse_message(self.tickers()[0]), container=True
                    )
                with gr.Column():
                    hw_rmse_md = gr.Markdown(
                        self.hw_rmse_message(self.tickers()[0]), container=True
                    )
            with gr.Row():
                ts_plot = self.timeseries_plot(self.tickers()[0])
            ticker_dropdown.change(
                ticker_change,
                inputs=[ticker_dropdown],
                outputs=[ts_plot, arima_rmse_md, hw_rmse_md],
            )

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

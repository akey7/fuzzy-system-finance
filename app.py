import os
import pytz
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error
import h5py
import matplotlib.pyplot as plt
import yaml
from s3_downloader import S3Downloader


class FSFinance:
    def __init__(self):
        """
        Instantiate the app class and read the csv that contains
        the data to be visualized.
        """
        load_dotenv()
        folder_path = "input"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder '{folder_path}' created")
        else:
            print(f"Folder '{folder_path}' already exists")
        bucket_name = os.getenv("PORTFOLIO_OPTIMIZATION_SPACE_NAME")
        time_series_bucket_name = os.getenv("TIME_SERIES_SPACE_NAME")
        hdf5_filename = "portfolio_optimization_plot_data.h5"
        optimization_metadata_filename = "optimization_metadata.yml"
        self.portfolio_optimization_plot_data_path = os.path.join(
            "input", hdf5_filename
        )
        optimization_metadata_path = os.path.join(
            "input", optimization_metadata_filename
        )
        all_forecasts_filename = "all_forecasts.csv"
        all_forecasts_local_filename = os.path.join("input", all_forecasts_filename)
        s3d = S3Downloader()
        s3d.download_file(
            bucket_name, hdf5_filename, self.portfolio_optimization_plot_data_path
        )
        s3d.download_file(
            bucket_name, optimization_metadata_filename, optimization_metadata_path
        )
        s3d.download_file(
            time_series_bucket_name,
            all_forecasts_filename,
            all_forecasts_local_filename,
        )
        with open(optimization_metadata_path, "r") as file:
            self.optimization_metadata = yaml.safe_load(file)
        self.df = pd.read_csv(all_forecasts_local_filename)
        self.df["pred_date"] = pd.to_datetime(self.df["pred_date"])

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
            ["pred_date", ticker, f"{ticker}_arima"]
        ].copy()
        select_df = select_df.melt(
            id_vars="pred_date",
            value_vars=[ticker, f"{ticker}_arima"],
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

    def optimization_metadata_markdown(self):
        tickers_and_weights = [
            f"{ticker}: {weight:.2f}%"
            for ticker, weight in self.optimization_metadata["optimum_portfolio"][
                "weights"
            ].items()
        ]
        date_from = self.optimization_metadata["date_updated"]["date_from"]
        date_to = self.optimization_metadata["date_updated"]["date_to"]
        annualized_return = self.optimization_metadata["optimum_portfolio"][
            "annualized_return"
        ]
        annualized_risk = self.optimization_metadata["optimum_portfolio"]["risk"]
        lines = [
            "## Max Sharpe Ratio Portfolio",
            "### Tickers and weights (shorting and leverage allowed):",
            ", ".join(tickers_and_weights),
            "### Historical Dates",
            f"{date_from} to {date_to}",
            "### Optimum portfolio annualized performance",
            f"Return: {annualized_return:.2f}%, Risk: {annualized_risk:.2f}%",
        ]
        return os.linesep.join(lines)

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
        ax.set_xlabel("Daily Risk (σ)")
        ax.set_ylabel("Daily Returns (%)")
        ax.set_title("Portfolio Optimization")
        ax.legend()
        return fig

    def calc_rmse(self, ticker):
        df0 = self.df.dropna()
        y_actual = df0[ticker]
        y_pred_arima = df0[f"{ticker}_arima"]
        y_pred_hw = df0[f"{ticker}_hw"]
        rmse_arima = root_mean_squared_error(y_actual, y_pred_arima)
        rmse_hw = root_mean_squared_error(y_actual, y_pred_hw)
        return rmse_arima, rmse_hw

    def arima_rmse_message(self, ticker):
        rmse_arima, _ = self.calc_rmse(ticker)
        rmse_arima_fmt = f"{rmse_arima:.2f}"
        return f"### ARIMA RMSE:{os.linesep}# {rmse_arima_fmt}"

    def run(self):
        """
        Run the Gradio app.
        """
        with gr.Blocks() as app:

            def ticker_change(choice):
                return (
                    self.timeseries_plot(choice),
                    self.arima_rmse_message(choice),
                )

            with gr.Row():
                with gr.Column():
                    portfolio_optimization_plot = gr.Plot(
                        self.plot_portfolio_optimization()
                    )

                with gr.Column():
                    with gr.Row():
                        optimization_metadata_md = gr.Markdown(
                            self.optimization_metadata_markdown()
                        )

                    with gr.Row():
                        gr.Markdown(
                            f"## Current and Future Day Models{os.linesep}### Select a ticker or index to view the model:"
                        )
                    with gr.Row():
                        ticker_dropdown = gr.Dropdown(
                            choices=self.tickers(),
                            label=None,
                            show_label=False,
                            value=self.tickers()[0],
                        )
            with gr.Row():
                ts_plot = self.timeseries_plot(self.tickers()[0])
                
            with gr.Row():
                with gr.Column():
                    arima_rmse_md = gr.Markdown(
                        self.arima_rmse_message(self.tickers()[0]), container=True
                    )
            
            ticker_dropdown.change(
                ticker_change,
                inputs=[ticker_dropdown],
                outputs=[ts_plot, arima_rmse_md],
            )

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

import os
import pytz
import gradio as gr
import pandas as pd
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error


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

    def tickers(self):
        """
        Return the sorted tickers available for visualization from the
        underlying DataFrame.
        """
        return sorted([ticker for ticker in self.df.columns if "pred" not in ticker])

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
        select_df = self.df[["pred_date", ticker, f"{ticker}_pred"]].copy()
        select_df = select_df.melt(
            id_vars="pred_date",
            value_vars=[ticker, f"{ticker}_pred"],
            var_name="Ticker",
            value_name="Adjusted Close ($)",
        )
        select_df["Ticker"] = select_df["Ticker"].apply(
            lambda x: (
                f"{x} (Actual)"
                if "pred" not in x
                else f"{x.replace('_pred', '')} (ARIMA Predicted)"
            )
        )
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
    
    def calc_mae(self, ticker):
        y_actual = self.df[ticker][:-1]
        y_pred = self.df[f"{ticker}_pred"][:-1]
        mae = mean_absolute_error(y_actual, y_pred)
        return mae
    
    def mae_message(self, ticker):
        mae_fmt = f"{self.calc_mae(ticker):.2f}"
        return f"### Mean Absolute Error (MAE):{os.linesep}# {mae_fmt}"

    def run(self):
        """
        Run the Gradio app.
        """
        with gr.Blocks() as app:

            def ticker_change(choice):
                return self.timeseries_plot(choice), self.mae_message(choice)

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("### Select a ticker or index:")
                    ticker_dropdown = gr.Dropdown(
                        choices=self.tickers(),
                        label=None,
                        show_label=False,
                        value=self.tickers()[0],
                    )
                with gr.Column():
                    mae_md = gr.Markdown(self.mae_message(self.tickers()[0]), container=True)
            with gr.Row():
                ts_plot = self.timeseries_plot(self.tickers()[0])
            ticker_dropdown.change(
                ticker_change, inputs=[ticker_dropdown], outputs=[ts_plot, mae_md]
            )

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

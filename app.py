import os
import gradio as gr
import pandas as pd


class FSFinance:
    def __init__(self):
        df_filename = os.path.join("input", "All Forecast 2025-02-24 to 2025-03-21.csv")
        self.df = pd.read_csv(df_filename)
        self.df["pred_date"] = pd.to_datetime(self.df["pred_date"])

    def tickers(self):
        return sorted([ticker for ticker in self.df.columns if "pred" not in ticker])

    def long_df(self, ticker):
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
                else f"{x.replace('_pred', '')} (Predicted)"
            )
        )
        select_df.rename(columns={"pred_date": "Date"}, inplace=True)
        return select_df

    def run(self):
        with gr.Blocks() as app:

            def ticker_change(choice):
                print(f"You selected {choice}")

            gr.Markdown("# fuzzy-system-finance")
            ticker_dropdown = gr.Dropdown(
                choices=self.tickers(),
                label="Select an option",
                value=self.tickers()[0],
            )
            ts_plot = gr.LinePlot(
                value=self.long_df("AAPL"),
                x="Date",
                y="Adjusted Close ($)",
                color="Ticker",
            )
            ticker_dropdown.change(ticker_change, inputs=[ticker_dropdown])

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

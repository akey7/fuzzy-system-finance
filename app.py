import os
import pytz
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

    def run(self):
        with gr.Blocks() as app:

            def ticker_change(choice):
                return self.timeseries_plot(choice)

            gr.Markdown("# fuzzy-system-finance")
            ticker_dropdown = gr.Dropdown(
                choices=self.tickers(),
                label="Select an option",
                value=self.tickers()[0],
            )
            ts_plot = self.timeseries_plot(self.tickers()[0])
            ticker_dropdown.change(
                ticker_change, inputs=[ticker_dropdown], outputs=[ts_plot]
            )

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

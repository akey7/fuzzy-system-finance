import os
import gradio as gr
import pandas as pd


class FSFinance:
    def __init__(self):
        df_filename = os.path.join("input", "All Forecast 2025-02-24 to 2025-03-21.csv")
        self.df = pd.read_csv(df_filename)
        self.df["pred_date"] = pd.to_datetime(self.df["pred_date"])
        self.df.set_index("pred_date", inplace=True)

    def tickers(self):
        return sorted([ticker for ticker in self.df.columns if "_pred" not in ticker])

    def run(self):
        with gr.Blocks() as app:

            def ticker_change(choice):
                print(f"You selected {choice}")

            gr.Markdown("# fuzzy-system-finance")
            ticker_dropdown = gr.Dropdown(
                choices=self.tickers(),
                label="Select an option",
                value=self.tickers()[0]
            )
            ticker_dropdown.change(ticker_change, inputs=[ticker_dropdown])

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

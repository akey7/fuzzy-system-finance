import os
import gradio as gr
import pandas as pd


class FSFinance:
    def __init__(self):
        df_filename = os.path.join("input", "All Forecast 2025-02-24 to 2025-03-21.csv")
        self.df = pd.read_csv(df_filename)
        self.df["pred_date"] = pd.to_datetime(self.df["pred_date"])
        self.df.set_index("pred_date", inplace=True)
        print(self.df.columns)

    def run(self):
        with gr.Blocks() as app:
            gr.Markdown("fuzzy-system-finance coming soon! Message two.")

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

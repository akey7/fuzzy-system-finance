import gradio as gr


class FSFinance:
    def run(self):
        with gr.Blocks() as app:
            gr.Markdown("fuzzy-system-finance coming soon! Message two.")

        app.launch()


if __name__ == "__main__":
    fsf = FSFinance()
    fsf.run()

import gradio as gr
from decontext import PaperContext, decontext


def load_sample_paper():
    with open("sample_full_text.json") as f:
        full_text_json_str = f.read()

    context = PaperContext.parse_raw(full_text_json_str)
    return context


paper_context = load_sample_paper()


def decontextualize(paper, selection):
    # return "Cheese"
    decontext_snippet = decontext(selection, paper_context)
    return decontext_snippet


def get_selection(selection: gr.SelectData):
    return selection.value


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            demo_paper = gr.Textbox(
                label="Paper",
                value=str(paper_context),
                interactive=True,
            )
            selected_text = gr.Textbox(label="Selected Text")
            submit_btn = gr.Button("Submit")
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Decontextualization",
                placeholder="Decontextualization goes here",
            )

    demo_paper.select(get_selection, None, outputs=selected_text)

    submit_btn.click(
        fn=decontextualize,
        inputs=[demo_paper, selected_text],
        outputs=output,
        api_name="decontextualize",
    )


if __name__ == "__main__":
    demo.launch(share=True)

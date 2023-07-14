from importlib import resources

from decontext.data_types import PaperSnippet
from .step import TemplatePipelineStep, SynthesisStep


class TemplateSynthStep(SynthesisStep, TemplatePipelineStep):
    def __init__(self):
        with resources.path("decontext.templates", "synth.yaml") as f:
            template_path = f
        super().__init__(model_name="text-davinci-003", template=template_path)

    def run(self, snippet: PaperSnippet):
        prompt = self.template.fill(
            {
                "questions": snippet.qae,
                "sentence": snippet.snippet,
            }
        )

        response = self.model(prompt)
        synth = self.model.extract_text(response)
        snippet.add_decontextualized_snippet(synth)
        snippet.add_cost(response.cost)

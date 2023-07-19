from importlib import resources

from decontext.data_types import PaperSnippet
from decontext.step.step import TemplatePipelineStep, QGenStep


class TemplateQGenStep(QGenStep, TemplatePipelineStep):
    def __init__(self):
        with resources.path("decontext.templates", "qgen.yaml") as f:
            template_path = f
        super().__init__(model_name="text-davinci-003", template=template_path)

    def run(self, snippet: PaperSnippet):
        prompt = self.template.fill(snippet.dict())
        response = self.model(prompt)
        text = self.model.extract_text(response)
        for line in text.strip().splitlines():
            question = line.lstrip(" -*")
            snippet.add_question(question=question)
        snippet.add_cost(response.cost)

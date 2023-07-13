from decontext.data_types import PaperSnippet
from decontext.step.step import TemplatePipelineStep, QGenStep


class TemplateQGenStep(QGenStep, TemplatePipelineStep):
    def __init__(self):
        super().__init__(
            model_name="text-davinci-003", template="templates/qgen.yaml"
        )

    def run(self, snippet: PaperSnippet):
        prompt = self.template.fill(snippet.dict())
        response = self.model(prompt)
        text = self.model.extract_text(response)
        for line in text.strip().splitlines():
            question = line.lstrip(" -*")
            snippet.add_question(question=question)
        snippet.add_cost(response.cost)

from importlib import resources
from typing import Optional

from decontext.cache import CacheState
from decontext.data_types import PaperSnippet
from decontext.step.step import TemplatePipelineStep, QGenStep


class TemplateQGenStep(TemplatePipelineStep, QGenStep):
    def __init__(self, cache_state: Optional[CacheState] = None):
        with resources.path("decontext.templates", "qgen.yaml") as f:
            template_path = f
        super().__init__(model_name="text-davinci-003", template=template_path, cache_state=cache_state)

    def _run(self, snippet: PaperSnippet, cache_state: Optional[CacheState] = None):
        prompt = self.template.fill(snippet.dict())
        response = self.model(prompt, cache_state=cache_state)
        text = self.model.extract_text(response)
        for line in text.strip().splitlines():
            question = line.lstrip(" -*")
            snippet.add_question(question=question)
        snippet.add_cost(response.cost)

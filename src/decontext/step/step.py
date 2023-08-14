from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from decontext.data_types import PaperSnippet
from decontext.model import load_model
from decontext.template import Template
from decontext.cache import CacheState


class PipelineStep:
    """Base class for Pipeline components."""

    name: str

    def __init__(self, cache_state: Optional[CacheState]) -> None:
        self.default_cache_state = cache_state if cache_state is not None else CacheState.NORMAL

    def run(self, snippet: PaperSnippet, cache_state: Optional[CacheState] = None):
        if cache_state is None:
            cache_state = self.default_cache_state

        self._run(snippet, cache_state)

    @abstractmethod
    def _run(self, snippet: PaperSnippet, cache_state: Optional[CacheState] = None):
        raise NotImplementedError()


class QGenStep(PipelineStep):
    name = "qgen"


class QAStep(PipelineStep):
    """Has to handle the logic of what to do with context."""

    name = "qa"


class SynthesisStep(PipelineStep):
    name = "synth"


class TemplatePipelineStep(PipelineStep):
    """Base class for steps that use templates"""

    def __init__(self, model_name: str, template: Union[str, Path], cache_state: Optional[CacheState] = None):
        """Initialize the Pipeline step by loading a model

        Args:
            model_name (str): name of the api model to use (e.g. 'text-davinci-003')
            template (str): template or path to template
        """
        super().__init__(cache_state=cache_state)
        self.model = load_model(model_name)
        self.template = Template(template)

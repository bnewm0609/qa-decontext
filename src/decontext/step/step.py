from decontext.data_types import PaperSnippet
from decontext.model import load_model
from decontext.template import Template


class PipelineStep:
    """Base class for Pipeline components."""

    name: str

    def run(self, snippet: PaperSnippet):
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

    def __init__(self, model_name: str, template: str):
        """Initialize the Pipeline step by loading a model

        Args:
            model_name (str): name of the api model to use (e.g. 'text-davinci-003')
            template (str): template or path to template
        """

        self.model = load_model(model_name)
        self.template = Template(template)

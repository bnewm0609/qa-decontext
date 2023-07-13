from typing import List, Optional, Tuple, Union

from pydantic import BaseModel


from decontext.data_types import (
    PaperContext,
    PaperSnippet,
)
from decontext.step.step import PipelineStep
from decontext.step.qa import TemplateRetrievalQAStep
from decontext.step.qgen import TemplateQGenStep
from decontext.step.synth import TemplateSynthStep
from decontext.logging import info


class Pipeline(BaseModel):
    # qgen: QGenStep
    # qa: QAStep
    # synth: SynthesisStep
    steps: List[PipelineStep]

    class Config:
        arbitrary_types_allowed = True


def decontext(
    snippet: str,
    context: Union[str, List[str], PaperContext, List[PaperContext]],
    # config: Union[Config, str, Path] = "configs/default.yaml",
    pipeline: Optional[Pipeline] = None,
    return_metadata: bool = False,
) -> Union[str, Tuple[str, PaperSnippet]]:
    """Decontextualizes the snippet using the given context according to the given config.

    Args:
        snippet: The text snippet to decontextualize.
        context: The context to incorporate in the decontextualization. This can be:
            * a string with the context.
            * a list of context strings (each item should be a paragraph).
            * a PaperContext object or list of PaperContext objects
        config: The configuration for the pipeline

    Returns:
        string with the decontextualized version of the snippet.

        if `return_metadata = True`, additionally return the intermediate results for each step of the pipeline
        as described above.
    """

    if pipeline is None:
        pipeline = Pipeline(
            steps=[
                TemplateQGenStep(),
                TemplateRetrievalQAStep(),
                TemplateSynthStep(),
            ]
        )

    # 2. Create the PaperSnippet object
    ps = PaperSnippet(snippet=snippet, context=context, qae=[])

    # 3. Runs each component of the pipeline
    for step in pipeline.steps:
        info(f"Running {step.name} > ")
        step.run(ps)

    if ps.decontextualized_snippet is None:
        decontext_snippet = ""
    else:
        decontext_snippet = ps.decontextualized_snippet

    if return_metadata:
        return decontext_snippet, ps
    else:
        return decontext_snippet

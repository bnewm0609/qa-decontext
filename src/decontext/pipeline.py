from typing import List, Optional, Tuple, Union

from pydantic import BaseModel

from decontext.data_types import (
    PaperContext,
    PaperSnippet,
)
from decontext.step.step import PipelineStep
from decontext.step.qa import TemplateRetrievalQAStep, TemplateFullTextQAStep
from decontext.step.qgen import TemplateQGenStep
from decontext.step.synth import TemplateSynthStep
from decontext.logging import info
from decontext.cache import CacheState


class Pipeline(BaseModel):
    steps: List[PipelineStep]

    class Config:
        arbitrary_types_allowed = True


# We don't want to instantiatate the pipelines here
class RetrievalQAPipeline(Pipeline):
    def __init__(self, **data):
        data["steps"] = [
            TemplateQGenStep(),
            TemplateRetrievalQAStep(),
            TemplateSynthStep(),
        ]


class FullTextQAPipeline(Pipeline):
    def __init__(self, **data):
        data["steps"] = [
            TemplateQGenStep(),
            TemplateFullTextQAStep(),
            TemplateSynthStep(),
        ]


def decontext(
    snippet: str,
    context: PaperContext,
    additional_contexts: Optional[List[PaperContext]] = None,
    pipeline: Optional[Pipeline] = None,
    return_metadata: bool = False,
    cache_states: Optional[Union[CacheState, List[Optional[CacheState]]]] = None,
) -> Union[str, Tuple[str, PaperSnippet]]:
    """Decontextualizes the snippet using the given context according to the given config.

    Args:
        snippet: The text snippet to decontextualize.
        context: The context to incorporate into the decontextualization. This context must include the snippet.
        additional_contexts: Additional context to use in the decontextualization (eg papers that are cited in
            the snippet).
        pipeline: The pipeline to run on the snippet.
        return_metadata: Flag for returning the PaperSnippet object with intermediate outputs. (See below).
        cache_states: The cache states to use for each step of the pipeline. If None, the default cache state
            is used. If a single CacheState is given, it is used for all steps. If a list of CacheStates is given,
            the ith CacheState is used for the ith step.

    Returns:
        string with the decontextualized version of the snippet.

        if `return_metadata = True`, additionally return the intermediate results for each step of the pipeline
        as described above.
    """

    # 1. Create the Pipepline
    if pipeline is None:
        pipeline = RetrievalQAPipeline()

    # 2. Create the PaperSnippet and PipelineData objects
    paper_snippet = PaperSnippet(
        snippet=snippet,
        context=context,
        additional_contexts=additional_contexts,
        qae=[],
    )

    # 3. Runs each component of the pipeline
    # 3.1. Handle the cache states
    if cache_states is None:
        cache_states = [None] * len(pipeline.steps)
    elif isinstance(cache_states, CacheState):
        cache_states = [cache_states] * len(pipeline.steps)
    for step, cache_state in zip(pipeline.steps, cache_states):
        info(f"Running {step.name} > ")
        # step.invalidate_cache = invalidate_cache
        step.run(paper_snippet, cache_state=cache_state)

    if paper_snippet.decontextualized_snippet is None:
        decontext_snippet = ""
    else:
        decontext_snippet = paper_snippet.decontextualized_snippet

    if return_metadata:
        return decontext_snippet, paper_snippet
    else:
        return decontext_snippet

from pathlib import Path
from typing import List, Optional, Tuple, Union

from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import BaseModel

from decontext import (
    EvidenceParagraph,
    Metadata,
    PaperContext,
    QuestionAnswerEvidence,
)

# eventually, we don't want to have to import *anything* from the experiment code because it has too much
# unnecessary overhead
from decontext.experiments.pipeline import (
    PaperSnippet as ExperimentPaperSnippet,
)
from decontext.experiments.pipeline import (
    question_answering,
    question_generation,
    run_retrieval,
    synthesize,
)
from decontext.experiments.utils import hash_strs


class PaperSnippet(BaseModel):
    idx: str = "0"
    snippet: str
    context: PaperContext
    qae: List[QuestionAnswerEvidence]
    decontextualized_snippet: Optional[str]

    def add_question(self, qid: str, question: str):
        self.qae.append(
            QuestionAnswerEvidence(
                qid=qid,
                question=question,
            )
        )

    def add_additional_paragraphs(
        self,
        qid: str,
        additional_paragraphs: List[str],
        section: Optional[str] = None,
        paper_id: Optional[str] = None,
    ):
        for qae in self.qae:
            if qae.qid == qid:
                if qae.evidence is None:
                    qae.evidence = []
                for additional_paragraph in additional_paragraphs:
                    qae.evidence.append(
                        EvidenceParagraph(
                            section=section,
                            paragraph=additional_paragraph,
                            paper_id=paper_id,
                        )
                    )

    def add_answer(self, qid: str, answer: str):
        for qae in self.qae:
            if qae.qid == qid:
                qae.answer = answer

    def add_decontextualized_snippet(self, decontextualized_snippet):
        self.decontextualized_snippet = decontextualized_snippet


class PipelineStep:
    name: str

    def run(self, snippet: PaperSnippet):
        raise NotImplementedError()


class QGenStep(PipelineStep):
    name = "qgen"


class RetreivalStep(PipelineStep):
    name = "qa_retrieval"


class QAStep(PipelineStep):
    """Has to handle the logic of what to do with context."""

    name = "qa"

    def __init__(self):
        pass


class SynthesisStep(PipelineStep):
    name = "synth"


class Pipeline(BaseModel):
    qgen: PipelineStep
    qa_retrieval: Optional[PipelineStep]
    qa: PipelineStep
    synth: PipelineStep

    class Config:
        arbitrary_types_allowed = True


DEFAULT_HYDRA_OVERRIDES = [
    "mode=predict",
    "model=pipeline-gpt4-retrieval-qa",
    # "model=pipeline-gpt4-qa",
    # "model.qa.data=template-chatgpt4-full-text-qa-2",
    "data=pipeline",
    "task=pipeline",
    "model.qgen.use_gold=False",
]


def init_hydra():
    OmegaConf.register_new_resolver("esc_slash", lambda x: x.replace("/", "-"))
    OmegaConf.register_new_resolver(
        "esc_period", lambda x: str(x).replace(".", "-")
    )
    # ignores trailing "/"
    OmegaConf.register_new_resolver(
        "extract_path",
        lambda path, num_segments: "-".join(
            path.strip("/").split("/")[-num_segments:]
        ),
    )
    OmegaConf.register_new_resolver("hash", hash_strs)
    # for now, use the experiment code
    # with initialize(version_base=None, config_path="../../configs"):
    initialize(version_base=None, config_path="../../configs")


class DefaultHydraStep(PipelineStep):
    def __init__(self):
        super().__init__()
        try:
            init_hydra()
        except ValueError:
            pass

        self.args = compose(
            config_name="config.yaml",
            overrides=DEFAULT_HYDRA_OVERRIDES,
        )
        Path(self.args.results_path).mkdir(exist_ok=True)

    def create_exp_snippet(self, snippet: PaperSnippet):
        assert snippet.context.paragraph_with_snippet is not None
        assert snippet.context.paragraph_with_snippet.section is not None
        exp_snippet = ExperimentPaperSnippet(
            idx="0",
            paper_id="0",
            title=snippet.context.title,
            abstract=snippet.context.abstract,
            full_text=[],
            section_header=snippet.context.paragraph_with_snippet.section,
            context=snippet.context.paragraph_with_snippet.paragraph,
            snippet=snippet.snippet,
            cited_ids=[],
        )

        # add in full text separetly if it's provided because the format is a bit differnent
        # {"section_name": "<section_name>", "paragraphs": ["<paragraph_1>", "<paragraph_2>", ...]}
        if snippet.context.full_text:
            exp_full_text = []
            for section in snippet.context.full_text:
                exp_full_text.append(
                    {
                        "section_name": section.section_name,
                        "paragraphs": section.paragraphs,
                    }
                )
            exp_snippet.full_text = exp_full_text  # type: ignore

        # add in anything that's already been computed
        # additional_paragraphs = {
        #     "section_name": "",
        # }

        questions = {}
        additional_paragraphs = {}
        answers = {}
        for question in snippet.qae:
            questions[question.qid] = question.question

            # if the additional paragraphs are added in the context, we should add them here for each question
            # otherwise, use retrieved evidence if it's there
            if snippet.context.additional_paragraphs:
                additional_paragraphs[question.qid] = [
                    ev.paragraph
                    for ev in snippet.context.additional_paragraphs
                ]
            elif question.evidence:
                additional_paragraphs[question.qid] = [
                    evidence.paragraph for evidence in question.evidence
                ]

            if question.answer:
                answers[question.qid] = question.answer

        exp_snippet.questions = questions
        exp_snippet.additional_paragraphs = additional_paragraphs
        exp_snippet.answers = answers

        return exp_snippet


class DefaultQGenStep(DefaultHydraStep, QGenStep):
    def run(self, snippet: PaperSnippet):
        exp_snippet = self.create_exp_snippet(snippet)
        question_generation(self.args, exp_snippet)
        for qid, question in exp_snippet.questions.items():
            snippet.add_question(qid=qid, question=question)


class DefaultQAStep(DefaultHydraStep, QAStep):
    # def __init__(self):
    #     super().__init__()
    #     if self.args.model.qa.retriever is not None:
    #         retriever =

    # def save_doc_for_retrieval(self, snippet: PaperSnippet):
    #     pass

    def run(self, snippet: PaperSnippet):
        exp_snippet = self.create_exp_snippet(snippet)
        if (
            snippet.context.full_text
            and self.args.model.qa.retriever is not None
        ):  # full text is in exp_snippet and this depend on the qa config
            run_retrieval(self.args, exp_snippet)
            for (
                qid,
                additional_paragraphs,
            ) in exp_snippet.additional_paragraphs.items():
                snippet.add_additional_paragraphs(qid, additional_paragraphs)
        # else:
        # No retrieval needed either because we're querying over the whole thing
        # or we're passed stuff in the context.
        # if snippet.context.additional_paragraphs:
        #     pass
        question_answering(self.args, exp_snippet)
        for qid, answer in exp_snippet.answers.items():
            snippet.add_answer(qid=qid, answer=answer)


class DefaultSynthesisStep(DefaultHydraStep, SynthesisStep):
    def run(self, snippet: PaperSnippet):
        exp_snippet = self.create_exp_snippet(snippet)
        synthesize(self.args, exp_snippet)
        snippet.add_decontextualized_snippet(
            exp_snippet.decontextualized_snippet
        )


def decontext(
    snippet: str,
    context: Union[str, List[str], PaperContext, List[PaperContext]],
    # config: Union[Config, str, Path] = "configs/default.yaml",
    return_metadata: bool = False,
) -> Union[str, Tuple[str, Metadata]]:
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
    pipeline = Pipeline(
        qgen=DefaultQGenStep(),
        # qa_retrieval=None, # DefaultRetrievalStep(),
        qa=DefaultQAStep(),
        synth=DefaultSynthesisStep(),
    )

    # 2. Create the PaperSnippet object
    ps = PaperSnippet(snippet=snippet, context=context, qae=[])

    # 3. Runs each component of the pipeline
    pipeline.qgen.run(ps)
    # if pipeline.qa_retrieval is not None:
    #     pipeline.qa_retrieval.run(ps)
    pipeline.qa.run(ps)
    pipeline.synth.run(ps)

    return ps

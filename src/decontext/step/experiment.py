"""The steps here tie in with the experiment code, which we want to avoid using.
"""
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from decontext.data_types import PaperSnippet
from decontext.step.step import PipelineStep, QAStep, QGenStep, SynthesisStep

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


class ExperimentStep(PipelineStep):
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
        section: str = ""
        if snippet.context.paragraph_with_snippet.section is not None:
            section = snippet.context.paragraph_with_snippet.section
        exp_snippet = ExperimentPaperSnippet(
            idx="0",
            paper_id="0",
            title=snippet.context.title,
            abstract=snippet.context.abstract,
            full_text=[],
            section_header=section,
            context=snippet.context.paragraph_with_snippet.paragraph,
            snippet=snippet.snippet,
            cited_ids=[],
        )

        # add in full text separetly if it's provided because the format is a bit differnent
        # {"section_name": "<section_name>", "paragraphs": ["<paragraph_1>", "<paragraph_2>", ...]}
        if snippet.context.full_text:
            exp_full_text = []
            for ctx_section in snippet.context.full_text:
                exp_full_text.append(
                    {
                        "section_name": ctx_section.section_name,
                        "paragraphs": ctx_section.paragraphs,
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


class ExperimentQGenStep(QGenStep, ExperimentStep):
    def run(self, snippet: PaperSnippet):
        exp_snippet = self.create_exp_snippet(snippet)
        question_generation(self.args, exp_snippet)
        for qid, question in exp_snippet.questions.items():
            snippet.add_question(qid=qid, question=question)


class ExperimentQAStep(QAStep, ExperimentStep):
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
                snippet.add_evidence_paragraphs(qid, additional_paragraphs)
        # else:
        # No retrieval needed either because we're querying over the whole thing
        # or we're passed stuff in the context.
        # if snippet.context.additional_paragraphs:
        #     pass
        question_answering(self.args, exp_snippet)
        for qid, answer in exp_snippet.answers.items():
            snippet.add_answer(qid=qid, answer=answer)


class DefaultSynthesisStep(SynthesisStep, ExperimentStep):
    def run(self, snippet: PaperSnippet):
        exp_snippet = self.create_exp_snippet(snippet)
        synthesize(self.args, exp_snippet)
        snippet.add_decontextualized_snippet(
            exp_snippet.decontextualized_snippet
        )

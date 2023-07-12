import json
import os
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional, Tuple, Union

from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import BaseModel
from shadow_scholar.app import pdod

from decontext.data_types import (
    Section,
    PaperContext,
    PaperSnippet,
)
from decontext.model import load_model
from decontext.template import Template

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


class DefaultQGenStep(DefaultHydraStep, QGenStep):
    def run(self, snippet: PaperSnippet):
        exp_snippet = self.create_exp_snippet(snippet)
        question_generation(self.args, exp_snippet)
        for qid, question in exp_snippet.questions.items():
            snippet.add_question(qid=qid, question=question)


class TemplatePipelineStep(PipelineStep):
    def __init__(self, name: str, model_name: str, template: str):
        self.model = load_model(model_name)
        self.template = Template(template)
        self.name = name

    def run(self, snippet: PaperSnippet):
        raise NotImplementedError


class TemplateQGenStep(TemplatePipelineStep):
    def __init__(self):
        super().__init__(
            "qgen", "text-davinci-003", "templates/qgen.yaml"
        )  # TODO use keywords
        self.retriever = "dense"

    def run(self, snippet: PaperSnippet):
        prompt = self.template.fill(snippet.dict())
        response = self.model(prompt)
        text = self.model.extract_text(response)
        for line in text.strip().splitlines():
            question = line.lstrip(" -*")
            snippet.add_question(question=question)
        snippet.add_cost(response.cost)


class TemplateRetrievalQAStep(TemplatePipelineStep):
    def __init__(self):
        super().__init__("qa", "gpt-4", "templates/qa_retrieval.yaml")

    def retrieve(self, paper_snippet: PaperSnippet):
        # TODO: cache these
        context = paper_snippet.context
        # 1. create the doc

        with ExitStack() as stack:
            doc_file = stack.enter_context(
                tempfile.NamedTemporaryFile(mode="w+", delete=False)
            )
            query_file = stack.enter_context(
                tempfile.NamedTemporaryFile(mode="w+", delete=False)
            )
            paper_retrieval_output_file = stack.enter_context(
                tempfile.NamedTemporaryFile(mode="w+", delete=False)
            )

            # with open(doc_path, "w") as f:
            for section in [
                Section(
                    section_name=context.title, paragraphs=[context.abstract]
                )
            ] + (context.full_text if context.full_text is not None else []):
                for para_i, paragraph in enumerate(section.paragraphs):
                    doc_file.write(
                        json.dumps(
                            {
                                "did": f"s{section.section_name}p{para_i}",
                                "text": paragraph,
                                "section": section.section_name,
                            }
                        )
                        + "\n"
                    )

            # 2. create the query
            for question in paper_snippet.qae:
                # with open(query_path, "a") as f:
                query_file.write(
                    json.dumps(
                        {"qid": question.qid, "text": question.question}
                    )
                    + "\n"
                )

            doc_file_name = doc_file.name
            query_file_name = query_file.name
            retrieval_output_file_name = paper_retrieval_output_file.name

        # 3. run retrieval
        try:
            ranker_kwargs = {"model_name_or_path": "facebook/contriever"}
            pdod.main.run_pdod(
                "dense",
                ranker_kwargs=ranker_kwargs,
                docs_path=doc_file_name,
                queries_path=query_file_name,
                output_path=retrieval_output_file_name,
            )

            # Extract the docs
            # breakpoint()
            with open(retrieval_output_file_name) as retrieval_output_file:
                docs = [
                    json.loads(line.strip()) for line in retrieval_output_file
                ]
            docs_by_qid = defaultdict(list)
            for doc in docs:
                docs_by_qid[doc["qid"]].append(doc["text"])
            for qid in docs_by_qid:
                # breakpoint()
                paper_snippet.add_evidence_paragraphs(
                    qid, docs_by_qid[qid][:3]
                )

        finally:
            os.remove(doc_file_name)
            os.remove(query_file_name)
            os.remove(retrieval_output_file_name)

        return paper_retrieval_output_file

    def run(self, snippet: PaperSnippet):
        self.retrieve(snippet)

        for question in snippet.qae:
            unique_evidence = set(
                [
                    ev.paragraph
                    for ev in (
                        question.evidence
                        if question.evidence is not None
                        else []
                    )
                    if (
                        ev.paragraph != snippet.context.abstract
                        and ev.paragraph
                        != snippet.context.paragraph_with_snippet
                    )
                ]
            )

            if snippet.context.paragraph_with_snippet is None:
                section_with_snippet = ""
                paragraph_with_snippet = ""
            else:
                para_w_snippet = snippet.context.paragraph_with_snippet
                section_with_snippet = (
                    ""
                    if para_w_snippet.section is None
                    else para_w_snippet.section
                )
                paragraph_with_snippet = (
                    ""
                    if para_w_snippet.paragraph is None
                    else para_w_snippet.paragraph
                )

            prompt = self.template.fill(
                {
                    "snippet": snippet.snippet,
                    "question": question.question,
                    "title": snippet.context.title,
                    "abstract": snippet.context.abstract,
                    "section_with_snippet": section_with_snippet,
                    "paragraph_with_snippet": paragraph_with_snippet,
                    "unique_evidence": list(unique_evidence),
                }
            )

            result = self.model(prompt)
            answer = self.model.extract_text(result)
            snippet.add_answer(qid=question.qid, answer=answer)
            snippet.add_cost(result.cost)


class TemplateFullTextQAStep(TemplatePipelineStep):
    def __init__(self):
        super().__init__("qa", "gpt-4", "templates/qa_fulltext.yaml")

    def run(self, snippet: PaperSnippet):
        for question in snippet.qae:
            prompt = self.template.fill(
                {
                    "snippet": snippet.snippet,
                    "question": question.question,
                    "full_text": str(snippet.context),
                }
            )

            response = self.model(prompt)
            answer = self.model.extract_text(response)
            snippet.add_answer(qid=question.qid, answer=answer)
            snippet.add_cost(response.cost)


class TemplateSynthStep(TemplatePipelineStep):
    def __init__(self):
        super().__init__("synth", "text-davinci-003", "templates/synth.yaml")

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


class DefaultQAStep(DefaultHydraStep, QAStep):
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
            qgen=DefaultQGenStep(),
            # qa_retrieval=None, # DefaultRetrievalStep(),
            qa=DefaultQAStep(),
            synth=DefaultSynthesisStep(),
        )

    # 2. Create the PaperSnippet object
    ps = PaperSnippet(snippet=snippet, context=context, qae=[])

    # 3. Runs each component of the pipeline
    print("QG > ")
    pipeline.qgen.run(ps)
    # if pipeline.qa_retrieval is not None:
    #     pipeline.qa_retrieval.run(ps)
    print("QA > ")
    pipeline.qa.run(ps)
    print("Synth > ")
    pipeline.synth.run(ps)

    if ps.decontextualized_snippet is None:
        decontext_snippet = ""
    else:
        decontext_snippet = ps.decontextualized_snippet

    if return_metadata:
        return decontext_snippet, ps
    else:
        return decontext_snippet

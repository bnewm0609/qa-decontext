import json
import os
import tempfile
from collections import defaultdict
from contextlib import ExitStack

from shadow_scholar.app import pdod

from decontext.data_types import PaperSnippet, Section
from decontext.step.step import QAStep, TemplatePipelineStep
from decontext.utils import none_check


class TemplateRetrievalQAStep(QAStep, TemplatePipelineStep):
    """Template step that does retrieval"""

    def __init__(self):
        super().__init__(
            model_name="gpt-4", template="templates/qa_retrieval.yaml"
        )

    def retrieve(self, paper_snippet: PaperSnippet):
        # TODO: cache these
        additional_contexts = none_check(paper_snippet.additional_contexts, [])
        contexts = [paper_snippet.context] + additional_contexts
        # 1. create the doc(s)

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

            for context in contexts:
                for section in [
                    Section(
                        section_name=context.title,
                        paragraphs=[context.abstract],
                    )
                ] + (none_check(context.full_text, [])):
                    for para_i, paragraph in enumerate(section.paragraphs):
                        doc_file.write(
                            json.dumps(
                                {
                                    "did": f"{context.id}.s{section.section_name}p{para_i}",
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
            with open(retrieval_output_file_name) as retrieval_output_file:
                docs = [
                    json.loads(line.strip()) for line in retrieval_output_file
                ]
            docs_by_qid = defaultdict(list)
            for doc in docs:
                docs_by_qid[doc["qid"]].append(doc["text"])
            for qid in docs_by_qid:
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
                    for ev in (none_check(question.evidence, []))
                    if (
                        ev.paragraph != snippet.context.abstract
                        and ev.paragraph != snippet.paragraph_with_snippet
                    )
                ]
            )

            paragraph_with_snippet = snippet.paragraph_with_snippet.paragraph
            section_with_snippet = none_check(
                snippet.paragraph_with_snippet.section, ""
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


class TemplateFullTextQAStep(QAStep, TemplatePipelineStep):
    """Runs the QA component of the decontextualization Pipeline using the whole context paper.

    All additional context papers are ignored.
    """

    def __init__(self):
        super().__init__(
            model_name="gpt-4", template="templates/qa_fulltext.yaml"
        )

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

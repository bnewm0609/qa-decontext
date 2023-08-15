import copy
import json
import os
import unittest

from decontext.cache import CacheState
from decontext.data_types import PaperSnippet, PaperContext, QuestionAnswerEvidence, Section
from decontext.step.qa import TemplateRetrievalQAStep


class TestTemplateRetrievalQaStep(unittest.TestCase):
    def setUp(self):
        # load the paper snippet
        self.qa_step = TemplateRetrievalQAStep()

        snippet_text = (
            "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new "
            "unannotated posts, labeling these posts with a probability of dogmatism according to the"
            " classifier (0=non-dogmatic, 1=dogmatic)."
        )

        with open("tests/fixtures/full_text_short.json") as f:
            full_text_json_str = f.read()

        self.context = PaperContext.parse_raw(full_text_json_str)

        self.paper_snippet = PaperSnippet(
            snippet=snippet_text,
            context=self.context,
            qae=[
                QuestionAnswerEvidence(
                    question="What does BOW+LING stand for?",
                    qid="0",
                )
            ],
        )

        self.using_github_actions = (
            "USING_GITHUB_ACTIONS" in os.environ and os.environ["USING_GITHUB_ACTIONS"] == "true"
        )

    def test_cache_override(self):
        # Create a snippet that is not in the cache to test instatiation with
        # `cache_state=CacheState.enforce_cache`

        qa_step = TemplateRetrievalQAStep(cache_state=CacheState.ENFORCE_CACHE)
        context = PaperContext(
            title="Sample title",
            abstract="Sample abstract",
            full_text=[
                Section(
                    section_name="Introduction", paragraphs=["Sample paragraph blah blah blah", "Second paragraph"]
                )
            ],
        )

        uncached_snippet = PaperSnippet(
            snippet="blah blah blah",  # this snippet is not in the cache
            context=context,
            qae=[
                QuestionAnswerEvidence(
                    question="What is the answer to life, the universe, and everything?",
                    qid="0",
                )
            ],
        )

        with self.assertRaises(ValueError):
            qa_step.run(uncached_snippet)

    def test_retrieval(self):
        if self.using_github_actions:
            self.skipTest(
                "Skipping test_retrieval because it requires downloading and running a huggingface model."
            )
        paper_snippet = copy.deepcopy(self.paper_snippet)
        self.qa_step.retrieve(paper_snippet)

        ev = [ev.paragraph for ev in paper_snippet.qae[0].evidence]

        with open("tests/fixtures/output_test_retrieval.json") as f:
            gold_ev = json.load(f)

        assert ev == gold_ev

    # def test_decontext_retrieval(self):
    #     pass
    # self.qa_step.retrieve(self.paper_snippet)

    # snippet = (
    #     "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new "
    #     "unannotated posts, labeling these posts with a probability of dogmatism according to the"
    #     " classifier (0=non-dogmatic, 1=dogmatic)."
    # )

    # with open("tests/fixtures/full_text.json") as f:
    #     full_text_json_str = f.read()

    # context = PaperContext.parse_raw(
    #     full_text_json_str
    # )  # parse_obj_as(PaperContext, full_text_dict)
    # # add the paragraph with the snippet bc that's important:
    # paragraph_with_snippet = None
    # for section in context.full_text:
    #     for paragraph in section.paragraphs:
    #         if snippet in paragraph:
    #             paragraph_with_snippet = EvidenceParagraph(
    #                 section=section.section_name, paragraph=paragraph
    #             )
    #             break
    #     if paragraph_with_snippet is not None:
    #         break
    # context.paragraph_with_snippet = paragraph_with_snippet

    # paper_snippet = decontext(snippet, context)

    # print(paper_snippet.decontextualized_snippet)

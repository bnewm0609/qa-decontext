import os
import unittest

from decontext.data_types import PaperSnippet, PaperContext
from decontext.step.qgen import TemplateQGenStep


class TestTemplateQGenStep(unittest.TestCase):
    def setUp(self):
        # load the paper snippet
        self.qgen_step = TemplateQGenStep()

        snippet_text = "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of "

        with open("tests/fixtures/full_text.json") as f:
            full_text_json_str = f.read()

        context = PaperContext.parse_raw(full_text_json_str)

        self.paper_snippet = PaperSnippet(
            snippet=snippet_text,
            context=context,
            qae=[],
        )

        self.using_github_actions = (
            "USING_GITHUB_ACTIONS" in os.environ and os.environ["USING_GITHUB_ACTIONS"] == "true"
        )

    # TODO: run this test in a way that doesn't require an openai key
    # def test_cache_invalidation(self):
    #     if self.using_github_actions:
    #         self.skipTest("Skipping test_cache_invalidation because it requires an openai key.")

    #     # TODO: create a fake cache file for testing this. For now just use a snippet that isn't used elsewhere.
    #     self.qgen_step.run(self.paper_snippet)

    #     questions = [qae.question for qae in self.paper_snippet.qae]

    #     self.qgen_step.invalidate_cache = True
    #     self.qgen_step.run(self.paper_snippet)

    #     questions_2 = [qae.question for qae in self.paper_snippet.qae]

    #     assert questions != questions_2

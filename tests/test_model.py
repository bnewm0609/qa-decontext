import io
import os
import sys
from unittest import mock

from decontext.model import load_model

# temporarily clear the environment variables
@mock.patch.dict(os.environ, {}, clear=True)
def test_warning_no_api_key():
    # temporarily redirect stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # loading the model should trigger the warning
    model = load_model("text-davinci-003")

    # reset stdout
    sys.stdout = sys.__stdout__
    assert (captured_output.getvalue() == "[WARNING] OPENAI_API_KEY not found in environment variables."
                "Set OPEN_API_KEY with your API key to use the OpenAI API.\n")
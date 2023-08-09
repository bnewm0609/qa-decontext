import io
import os
import sys
import json
from unittest import mock

from decontext.data_types import OpenAIChatMessage
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
    assert (
        captured_output.getvalue() == "[WARNING] OPENAI_API_KEY not found in environment variables."
        "Set OPEN_API_KEY with your API key to use the OpenAI API.\n"
    )


def test_get_key():
    model = load_model("text-davinci-003")
    params = {"top_p": 0.1, "prompt": "This is a test prompt", "user": "ignored"}
    key = model.get_key(params)
    assert key == json.dumps({"top_p": 0.1, "prompt": "This is a test prompt"}, sort_keys=True)

    model = load_model("gpt-4")
    messages = [OpenAIChatMessage(role="user", content="test content").dict()]
    params = {"top_p": 0.1, "messages": messages, "user": "ignored"}
    key = model.get_key(params)
    assert key == '{"messages": [{"content": "test content", "role": "user"}], "top_p": 0.1}'

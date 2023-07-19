from typing import Optional, get_type_hints

from decontext.utils import none_check


def test_none_check() -> None:
    x: Optional[str] = "x"
    # breakpoint()
    module_type_hints = get_type_hints(__import__(__name__))
    print(module_type_hints.get("x"))

    assert none_check(x, "") == "x"

    x = None
    assert none_check(x, "") == ""

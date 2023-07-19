import base64
import hashlib
from typing import TypeVar, Optional


def hash_strs(*strs, lim=10) -> str:
    """Hash a list of strs.

    Args:
        strs (list[str]): the strings to hash.
        lim (int): how many characters the output string should have.

    Returns:
        A hash `lim` characters long.
    """
    if isinstance(strs[0], int):
        lim, *str_tup = strs
        strs = tuple(str_tup)
    inpt = "".join([f"s{i}:{str(x)}" for i, x in enumerate(strs)])
    digest = hashlib.md5(inpt.encode("UTF-8")).digest()
    return base64.b64encode(digest).decode("UTF-8")[:lim]


T = TypeVar("T")


def none_check(value: Optional[T], default: T) -> T:
    """_summary_

    Args:
        value (Any): the value that is potentially None

    Returns:
        Any: returns the value or a new instance of the type of the value if the value is None
    """
    return value if value is not None else default

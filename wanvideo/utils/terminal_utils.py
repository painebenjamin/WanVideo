from typing import Any

from .import_utils import termcolor_available

__all__ = [
    "maybe_use_termcolor",
]


def maybe_use_termcolor(text: str, *args: Any, **kwargs: Any) -> str:
    """
    If termcolor is available, colorizes the text.
    Otherwise, returns the text unchanged.
    """
    if termcolor_available():
        from termcolor import (
            colored,
        )  # type: ignore[import-untyped,import-not-found,unused-ignore]

        return colored(text, *args, **kwargs)
    return text

from collections.abc import Iterable, Iterator
from typing import Any

from .import_utils import tqdm_available

__all__ = [
    "maybe_use_tqdm",
]


def maybe_use_tqdm(
    iterable: Iterable[Any],
    use_tqdm: bool = True,
    desc: str | None = None,
    total: int | None = None,
    unit: str = "it",
    unit_scale: bool = False,
    unit_divisor: int = 1000,
) -> Iterator[Any]:
    """
    Return the iterable or wrap it in a tqdm if use_tqdm is True.

    :param iterable: The iterable to return.
    :param use_tqdm: Whether to wrap the iterable in a tqdm.
    :param desc: The description to display.
    :param total: The total number of items.
    :param unit: The unit to display.
    :param unit_scale: Whether to scale the unit.
    :param unit_divisor: The unit divisor.
    :return: The iterable or tqdm wrapped iterable.
    """
    if use_tqdm and tqdm_available():
        from tqdm import tqdm

        yield from tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            unit_scale=unit_scale,
            unit_divisor=unit_divisor,
        )
    else:
        yield from iterable

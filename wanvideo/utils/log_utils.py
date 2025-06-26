import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from math import modf
from typing import Sequence

from .terminal_utils import maybe_use_termcolor

__all__ = [
    "logger",
    "debug_logger",
    "log_duration",
    "human_size",
    "human_duration",
    "reduce_units",
]

logger = logging.getLogger("wanvideo")
logger.setLevel(logging.WARNING)


class ColoredLoggingFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return {
            "CRITICAL": maybe_use_termcolor(formatted, "red", attrs=["bold"]),
            "ERROR": maybe_use_termcolor(formatted, "red"),
            "WARNING": maybe_use_termcolor(formatted, "yellow"),
            "INFO": maybe_use_termcolor(formatted, "green"),
            "DEBUG": maybe_use_termcolor(formatted, "cyan"),
            "NOTSET": formatted,
        }.get(record.levelname.upper(), formatted)


@contextmanager
def debug_logger() -> Iterator[logging.Logger]:
    """
    Returns a logger configured for debugging.
    """
    initial_level = logger.level
    initial_handlers = logger.handlers[:]

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = ColoredLoggingFormatter(
        "%(asctime)s [%(name)s] %(levelname)s - (%(filename)s:%(lineno)s): %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        yield logger
    finally:
        logger.setLevel(initial_level)
        logger.handlers.clear()
        for h in initial_handlers:
            logger.addHandler(h)


def reduce_units(
    value: int | float,
    units: Sequence[str | tuple[str, int | float]],
    base: int | float = 1000,
) -> tuple[float, str]:
    """
    Reduce a value to the smallest unit possible.

    >>> reduce_units(4e9, ["bytes/s", "kb/s", "mb/s", "gb/s"])
    (4.0, 'gb/s')
    """
    try:
        unit = units[0]
    except IndexError:
        raise ValueError("At least one unit must be provided.")

    for unit_or_tuple in units:
        if isinstance(unit_or_tuple, tuple):
            unit, unit_base = unit_or_tuple
        else:
            unit = unit_or_tuple
            unit_base = base
        if value < unit_base:
            break
        value /= unit_base
    return value, unit  # type: ignore[return-value]


def human_size(num_bytes: int | float, base_2: bool = False, precision: int = 2) -> str:
    """
    Convert a number of bytes to a human-readable string.

    >>> human_size(1000)
    '1.00 KB'
    >>> human_size(1000**3)
    '1.00 GB'
    >>> human_size(1024, base_2=True)
    '1.00 KiB'
    >>> human_size(1024**3, base_2=True)
    '1.00 GiB'
    """
    if base_2:
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        divisor = 1024.0
    else:
        units = ["B", "KB", "MB", "GB", "TB"]
        divisor = 1000.0

    reduced_bytes, unit = reduce_units(num_bytes, units, base=divisor)

    return f"{reduced_bytes:.{precision}f} {unit}"


def human_duration(
    duration_s: int | float,
    precision: int | None = None,
) -> str:
    """
    Convert a number of seconds to a human-readable string.
    Decimal precision is variable.

    Value < 1 second:
        Nanoseconds, microseconds, and milliseconds are reported as integers.
    1 second < value < 1 minute:
        Seconds are reported as floats with one decimal place.
    1 minute < value < 1 hour:
        Reported as minutes and seconds in the format "<x> m <y> s" with no decimal places.
    1 hour < value < 1 day:
        Reported as hours and minutes in the format "<x> h <y> m <z> s" with no decimal places.
    1 day < value:
        Reported as days and hours in the format "<x> d <y> h <z> m <zz> s" with no decimal places.

    >>> human_duration(0.00001601)
    '16 µs'
    >>> human_duration(1.5)
    '1.5 s'
    >>> human_duration(65)
    '1 m 5 s'
    >>> human_duration(3665)
    '1 h 1 m 5 s'
    >>> human_duration(90065)
    '1 d 1 h 1 m 5 s'
    """
    # First set the duration to nanoseconds
    duration_s *= 1e9
    units = ["ns", "µs", "ms", "s", "m", "h", "d"]
    bases = [1e3, 1e3, 1e3, 60, 60, 24, 1000]
    reduced_seconds, unit = reduce_units(
        duration_s,
        list(zip(units, bases)),
        base=1000,
    )
    if unit in ["d", "h", "m"]:
        # Split the seconds into a whole part and a fractional part
        fractional, whole = modf(reduced_seconds)
        whole_formatted = f"{whole:.0f} {unit}"
        if fractional == 0:
            return whole_formatted
        # Return the fractional part to seconds
        if unit in ["d", "h", "m"]:
            fractional *= 60
        if unit in ["d", "h"]:
            fractional *= 60
        if unit == "d":
            fractional *= 24
        return " ".join([whole_formatted, human_duration(fractional, precision=0)])
    else:
        if unit in ["ns", "µs", "ms"] and precision is None:
            precision = 1 if reduced_seconds < 10 else 0
        elif unit == "s" and precision is None:
            precision = 1
        return f"{reduced_seconds:.{precision}f} {unit}"


@contextmanager
def log_duration(name: str | None) -> Iterator[None]:
    """
    Context manager to log the duration of a block of code.
    """
    start_time = time.perf_counter()
    if name is None:
        name = "execution"

    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"{name} took {human_duration(duration)}")

__all__ = [
    "tqdm_available",
    "termcolor_available",
    "flash_attn_available",
    "flash_attn_2_available",
    "flash_attn_3_available",
]

TQDM_AVAILABLE: bool | None = None
TERMCOLOR_AVAILABLE: bool | None = None
FLASH_ATTN_3_AVAILABLE: bool | None = None
FLASH_ATTN_2_AVAILABLE: bool | None = None


def flash_attn_3_available() -> bool:
    """
    :return: whether flash attention 3 is available
    """
    global FLASH_ATTN_3_AVAILABLE
    if FLASH_ATTN_3_AVAILABLE is None:
        try:
            import flash_attn_interface  # type: ignore[import-untyped,import-not-found,unused-ignore]

            FLASH_ATTN_3_AVAILABLE = flash_attn_interface is not None
        except ModuleNotFoundError:
            FLASH_ATTN_3_AVAILABLE = False
    return FLASH_ATTN_3_AVAILABLE


def flash_attn_2_available() -> bool:
    """
    :return: whether flash attention 2 is available
    """
    global FLASH_ATTN_2_AVAILABLE
    if FLASH_ATTN_2_AVAILABLE is None:
        try:
            import flash_attn  # type: ignore[import-untyped,import-not-found,unused-ignore]

            FLASH_ATTN_2_AVAILABLE = flash_attn is not None
        except ModuleNotFoundError:
            FLASH_ATTN_2_AVAILABLE = False
    return FLASH_ATTN_2_AVAILABLE


def flash_attn_available() -> bool:
    """
    :return: whether flash attention is available
    """
    return flash_attn_3_available() or flash_attn_2_available()


def tqdm_available() -> bool:
    """
    Return whether tqdm is available.

    :return: Whether tqdm is available.
    """
    global TQDM_AVAILABLE
    if TQDM_AVAILABLE is None:
        try:
            import tqdm  # type: ignore[import-untyped,import-not-found,unused-ignore]

            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False
    return TQDM_AVAILABLE


def termcolor_available() -> bool:
    """
    Return whether termcolor is available.

    :return: Whether termcolor is available.
    """
    global TERMCOLOR_AVAILABLE
    if TERMCOLOR_AVAILABLE is None:
        try:
            import termcolor  # type: ignore[import-untyped,import-not-found,unused-ignore]

            TERMCOLOR_AVAILABLE = True
        except ImportError:
            TERMCOLOR_AVAILABLE = False
    return TERMCOLOR_AVAILABLE

from math import sqrt
from collections.abc import Iterator
from typing import Any
from contextlib import contextmanager

import torch
import torch.nn.functional as F


__all__ = [
    "torch_dtype_from_string",
    "get_torch_dtype",
    "get_torch_device",
    "finite_clamp",
    "float16_finite_clamp",
    "pos_interpolate",
    "no_init_weights",
]


def torch_dtype_from_string(torch_type: str) -> torch.dtype:
    """
    Converts a string to a torch DType.
    """
    try:
        return {
            "complex128": torch.complex128,
            "cdouble": torch.complex128,
            "complex": torch.complex64,
            "complex64": torch.complex64,
            "cfloat": torch.complex64,
            "cfloat64": torch.complex64,
            "cf64": torch.complex64,
            "double": torch.float64,
            "float64": torch.float64,
            "fp64": torch.float64,
            "float": torch.float32,
            "full": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
            "float8": torch.float8_e4m3fn,
            "float8_e4m3": torch.float8_e4m3fn,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "fp84": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            "fp85": torch.float8_e5m2,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "short": torch.int16,
            "int": torch.int32,
            "int32": torch.int32,
            "long": torch.int64,
            "int64": torch.int64,
            "bool": torch.bool,
            "bit": torch.bool,
            "1": torch.bool,
        }[torch_type[6:] if torch_type.startswith("torch.") else torch_type]
    except KeyError:
        raise ValueError(f"Unknown torch type '{torch_type}'")


def get_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """
    Gets the torch data type from a string.
    """
    if dtype is None:
        return torch.float32
    elif isinstance(dtype, str):
        return torch_dtype_from_string(dtype)
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise TypeError(f"Expected str or torch.dtype, got {type(dtype)}")

def get_torch_device(device: str | torch.device | None | int) -> torch.device:
    """
    Gets the torch device from a string or torch.device.
    """
    if device is None:
        return torch.device("cpu")
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, int):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError(f"Expected str, torch.device, int, or None, got {type(device)}")

def finite_clamp(x: torch.Tensor, offset: float = 1e4) -> torch.Tensor:
    """
    Clamps a tensor if it contains infinite values.

    Uses `torch.finfo` to get the maximum value of the tensor's dtype and
    subtracts the offset to get the clamp value.
    """
    import torch

    clamp_value = torch.finfo(x.dtype).max - offset
    return torch.clamp(x, -clamp_value, clamp_value)


def float16_finite_clamp(x: torch.Tensor, offset: float = 1e4) -> torch.Tensor:
    """
    Clamps a tensor if it is float16 and contains infinite values.

    Uses `torch.finfo` to get the maximum value of the tensor's dtype and
    subtracts the offset to get the clamp value.
    """
    import torch

    if x.dtype is torch.float16:
        return finite_clamp(x, offset)
    return x


def pos_interpolate(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Interpolate the positional embedding to match the sequence length.
    :param pos: [1, L, C] or [B, L, C].
    :param seq_len: target sequence length.
    :return: [B, seq_len, C].
    """
    if pos.size(1) == seq_len:
        return pos
    else:
        src_grid = int(sqrt(pos.size(1)))
        tar_grid = int(sqrt(seq_len))
        n = pos.size(1) - src_grid * src_grid
        return torch.cat(
            [
                pos[:, :n],
                F.interpolate(
                    pos[:, n:]
                    .float()
                    .reshape(1, src_grid, src_grid, -1)
                    .permute(0, 3, 1, 2),
                    size=(tar_grid, tar_grid),
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .transpose(1, 2),
            ],
            dim=1,
        )

@contextmanager
def no_init_weights() -> Iterator[None]:
    """
    Context manager to globally disable weight initialization to speed up loading large models.
    """
    torch_init_functions = {
        "uniform_": torch.nn.init.uniform_,
        "normal_": torch.nn.init.normal_,
        "trunc_normal_": torch.nn.init.trunc_normal_,
        "constant_": torch.nn.init.constant_,
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
        "kaiming_normal_": torch.nn.init.kaiming_normal_,
        "uniform": torch.nn.init.uniform,
        "normal": torch.nn.init.normal,
        "xavier_uniform": torch.nn.init.xavier_uniform,
        "xavier_normal": torch.nn.init.xavier_normal,
        "kaiming_uniform": torch.nn.init.kaiming_uniform,
        "kaiming_normal": torch.nn.init.kaiming_normal,
    }

    def _skip_init(*args: Any, **kwargs: Any) -> None:
        pass

    for name in torch_init_functions.keys():
        setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        # Restore the original initialization functions
        for name, init_func in torch_init_functions.items():
            setattr(torch.nn.init, name, init_func)

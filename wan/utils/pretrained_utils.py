from __future__ import annotations

from typing import Any, TypeVar

import torch
import torch.nn as nn
from diffusers.models.model_loading_utils import load_state_dict

from .torch_utils import get_torch_device, get_torch_dtype, no_init_weights

__all__ = [
    "PretrainedMixin",
]

T = TypeVar("T", bound=nn.Module)


class PretrainedMixin(nn.Module):
    """
    A mixin class for loading pretrained models in PyTorch.

    This class provides a method to load a pretrained model from a single file.
    It is designed to be used with PyTorch models that inherit from `nn.Module`.
    """

    @classmethod
    def from_single_file(
        cls: type[T],
        path: str,
        device: str | int | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Load a pretrained model from a single file.
        :param path: Path to the model file.
        :param device: Device to load the model on.
        :param dtype: Data type to load the model in.
        :param kwargs: Additional keyword arguments.
        :return: An instance of the model.
        """
        with no_init_weights():
            model = cls(**kwargs)
            model.load_state_dict(
                load_state_dict(path),
                strict=True,
            )
            model.to(
                device=get_torch_device(device),
                dtype=None if dtype is None else get_torch_dtype(dtype),
            )

        return model

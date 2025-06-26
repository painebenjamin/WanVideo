import torch

from .torch_utils import get_torch_dtype

__all__ = [
    "reschedule_noise",
    "make_noise",
]


def reschedule_noise(
    noise: torch.Tensor,
    window_size: int,
    window_stride: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Reschedules noise scross animation frames for more consistent diffusion-based animations
    See https://arxiv.org/abs/2310.15169
    """
    import torch

    _, _, frames, _, _ = noise.shape

    for frame_index in range(window_size, frames, window_stride):
        start_index = max(0, frame_index - window_size)
        end_index = min(frames, start_index + window_stride)
        window_length = end_index - start_index

        if window_length == 0:
            break

        list_indices = list(range(start_index, end_index))
        indices = torch.LongTensor(list_indices).to(noise.device)
        shuffled_indices = indices[torch.randperm(window_length, generator=generator)]

        current_start = frame_index
        current_end = min(frames, current_start + window_length)

        if current_end == current_start + window_length:
            # Fits perfectly in window
            noise[:, :, current_start:current_end] = noise[:, :, shuffled_indices]
        else:
            # Need to wrap around
            prefix_length = current_end - current_start
            shuffled_indices = shuffled_indices[:prefix_length]
            noise[:, :, current_start:current_end] = noise[:, :, shuffled_indices]

    return noise


def make_noise(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    frames: int,
    reschedule_window_size: int | None = None,
    reschedule_window_stride: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | str = "float32",
) -> torch.Tensor:
    """
    Creates a noise tensor with the specified dimensions and optionally reschedules it.
    :param batch_size: The batch size.
    :param channels: The number of channels.
    :param height: The height of the tensor.
    :param width: The width of the tensor.
    :param frames: The number of frames.
    :param reschedule_window_size: The size of the window for rescheduling noise.
    :param reschedule_window_stride: The stride of the window for rescheduling noise.
    :param generator: Optional random number generator.
    :param device: The device to create the tensor on.
    :param dtype: The data type of the tensor.
    :return: A tensor of random noise.
    """
    noise = torch.randn(
        (batch_size, channels, frames, height, width),
        generator=generator,
        device="cpu" if device is None else device,
        dtype=get_torch_dtype(dtype),
    )

    if reschedule_window_size and reschedule_window_stride:
        noise = reschedule_noise(
            noise, reschedule_window_size, reschedule_window_stride, generator=generator
        )

    return noise

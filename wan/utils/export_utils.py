import shutil
import tempfile

import torch
from imageio import get_writer
from torchvision.utils import make_grid  # type: ignore [import-untyped]


def write_video(
    video: torch.Tensor,
    path: str,
    nrow: int = 1,
    fps: int = 30,
    suffix: str = ".mp4",
    codec: str = "libx264",
    quality: int = 8,
    normalize: bool = True,
    value_range: tuple = (-1.0, 1.0),
) -> None:
    """
    Write a video tensor to a file.
    :param video: A tensor of shape (B, C, T, H, W) representing the video.
    :param path: The file path to save the video.
    :param nrow: Number of frames to display in a row.
    :param fps: Frames per second for the video.
    :param suffix: File suffix for the video format.
    :param codec: Codec to use for encoding the video.
    :param quality: Quality of the video encoding.
    :param normalize: Whether to normalize the video tensor.
    :param value_range: Tuple indicating the range of values in the video tensor.
    """
    if video.dim() == 4 and video.shape[0] == 3:
        # This is likely [C, T, H, W] format, convert to [B, C, T, H, W]
        video = video.unsqueeze(0)

    video = video.clamp(value_range[0], value_range[1])
    video = torch.stack(
        [
            make_grid(frame, nrow=nrow, normalize=normalize, value_range=value_range)
            for frame in video.unbind(2)
        ],
        dim=1,
    )
    video = video.permute(1, 2, 3, 0)
    video = (video * 255).to(torch.uint8).cpu()

    with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
        temp_path = temp_file.name
        with get_writer(temp_path, fps=fps, codec=codec, quality=quality) as writer:
            for frame_array in video.numpy():
                writer.append_data(frame_array)  # type: ignore[attr-defined]

        shutil.copy(temp_path, path)

from math import ceil

__all__ = [
    "sliding_1d_windows",
    "sliding_1d_window_count",
    "sliding_2d_windows",
    "sliding_2d_window_count",
    "sliding_3d_windows",
    "sliding_3d_window_count",
]


def sliding_1d_windows(
    length: int, window_size: int, window_stride: int
) -> list[tuple[int, int]]:
    """
    Gets windows over a length using a square tile.

    :param length: The length of the area.
    :param window_size: The size of the tile.
    :param window_stride: The stride of the tile.
    """
    coords: list[tuple[int, int]] = []
    for start in range(0, length - window_size + 1, window_stride):
        coords.append((start, start + window_size))
    if (length - window_size) % window_stride != 0:
        coords.append((length - window_size, length))
    return coords


def sliding_1d_window_count(length: int, window_size: int, window_stride: int) -> int:
    """
    Calculate the number of tiles needed to cover a length.

    :param length: The length of the area.
    :param window_size: The size of the tile.
    :param window_stride: The stride of the tile.
    :return: The number of tiles needed to cover the area.
    """
    return ceil((length - window_size) / window_stride + 1)


def sliding_2d_windows(
    height: int,
    width: int,
    tile_size: int | tuple[int, int],
    tile_stride: int | tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """
    Gets windows over a height/width using a square tile.

    :param height: The height of the area.
    :param width: The width of the area.
    :param tile_size: The size of the tile.
    :param tile_stride: The stride of the tile.

    :return: A list of tuples representing the windows in the format (top, bottom, left, right).
    """
    if isinstance(tile_size, tuple):
        tile_width, tile_height = tile_size
    else:
        tile_width = tile_height = int(tile_size)

    if isinstance(tile_stride, tuple):
        tile_stride_width, tile_stride_height = tile_stride
    else:
        tile_stride_width = tile_stride_height = int(tile_stride)

    height_list = list(range(0, height - tile_height + 1, tile_stride_height))
    if (height - tile_height) % tile_stride_height != 0:
        height_list.append(height - tile_height)

    width_list = list(range(0, width - tile_width + 1, tile_stride_width))
    if (width - tile_width) % tile_stride_width != 0:
        width_list.append(width - tile_width)

    coords: list[tuple[int, int, int, int]] = []
    for height in height_list:
        for width in width_list:
            coords.append((height, height + tile_height, width, width + tile_width))

    return coords


def sliding_2d_window_count(
    height: int,
    width: int,
    tile_size: int | tuple[int, int],
    tile_stride: int | tuple[int, int],
) -> int:
    """
    Calculate the number of tiles needed to cover a rectangular area.

    :param height: The height of the area.
    :param width: The width of the area.
    :param tile_size: The size of the tile.
    :param tile_stride: The stride of the tile.
    :return: The number of tiles needed to cover the area.
    """
    from math import ceil

    if isinstance(tile_size, tuple):
        tile_width, tile_height = tile_size
    else:
        tile_width = tile_height = int(tile_size)

    if isinstance(tile_stride, tuple):
        tile_stride_width, tile_stride_height = tile_stride
    else:
        tile_stride_width = tile_stride_height = int(tile_stride)

    return ceil((height - tile_height) / tile_stride_height + 1) * ceil(
        (width - tile_width) / tile_stride_width + 1
    )


def sliding_3d_windows(
    frames: int,
    height: int,
    width: int,
    frame_window_size: int,
    frame_window_stride: int,
    tile_size: int | tuple[int, int],
    tile_stride: int | tuple[int, int],
    temporal_first: bool = True,
) -> list[tuple[int, int, int, int, int, int]]:
    """
    Gets windows over a video using a 3D tile.

    :param frames: The number of frames in the video.
    :param height: The height of the video.
    :param width: The width of the video.
    :param frame_window_size: The size of the frame window.
    :param frame_window_stride: The stride of the frame window.
    :param tile_size: The size of the tile.
    :param tile_stride: The stride of the tile.
    """
    temporal_windows = sliding_1d_windows(
        frames, frame_window_size, frame_window_stride
    )
    spatial_windows = sliding_2d_windows(height, width, tile_size, tile_stride)
    spatiotemporal_windows: list[tuple[int, int, int, int, int, int]] = []

    if temporal_first:
        for temporal_window in temporal_windows:
            for spatial_window in spatial_windows:
                spatiotemporal_windows.append((*temporal_window, *spatial_window))
    else:
        for spatial_window in spatial_windows:
            for temporal_window in temporal_windows:
                spatiotemporal_windows.append((*spatial_window, *temporal_window))

    return spatiotemporal_windows


def sliding_3d_window_count(
    frames: int,
    height: int,
    width: int,
    frame_window_size: int,
    frame_window_stride: int,
    tile_size: int | tuple[int, int],
    tile_stride: int | tuple[int, int],
) -> int:
    """
    Calculate the number of tiles needed to cover a video.

    :param frames: The number of frames in the video.
    :param height: The height of the video.
    :param width: The width of the video.
    :param frame_window_size: The size of the frame window.
    :param frame_window_stride: The stride of the frame window.
    :param tile_size: The size of the tile.
    :param tile_stride: The stride of the tile.
    """
    temporal_windows = sliding_1d_window_count(
        frames, frame_window_size, frame_window_stride
    )
    spatial_windows = sliding_2d_window_count(height, width, tile_size, tile_stride)
    return temporal_windows * spatial_windows

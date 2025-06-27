import torch

__all__ = [
    "get_optimized_cfg_alpha",
]


def get_optimized_cfg_alpha(
    positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    """
    Calculates an optimized scalar to correct for inaccuracies in
    expected flow velocity estimation as proposed by Weiche Fan
    et. al in CFG-Zero*

    :param positive: Positive guidance embedding
    :param negative: Negative guidance embedding
    :return: Optimized scalar
    :see https://arxiv.org/abs/2503.18886:
    """
    dot_product = torch.sum(positive * negative, dim=1, keepdim=True)
    squared_norm = torch.sum(negative**2, dim=1, keepdim=True) + 1e-8
    st_star = dot_product / squared_norm
    return st_star

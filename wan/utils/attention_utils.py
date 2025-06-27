import torch

from .import_utils import (
    flash_attn_2_available,
    flash_attn_3_available,
    flash_attn_available,
)
from .torch_utils import get_torch_dtype

__all__ = [
    "flash_attention",
    "attention",
]


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: str | torch.dtype = "bfloat16",
    version: int | None = None,
) -> torch.Tensor:
    """
    :param q: query tensor, shape [batch, seq_len_q, dim]
    :param k: key tensor, shape [batch, seq_len_k, dim]
    :param v: value tensor, shape [batch, seq_len_k, dim]
    :param q_lens: query sequence lengths, shape [batch]
    :param k_lens: key sequence lengths, shape [batch]
    :param dropout_p: dropout probability
    :param softmax_scale: scale factor for softmax
    :param q_scale: scale factor for query
    :param causal: whether to use causal attention
    :param window_size: window size for local attention
    :param deterministic: whether to use deterministic attention
    :param dtype: data type for computation
    :param version: version of flash attention, 2 or 3
    :return: output tensor, shape [batch, seq_len_q, dim]
    """
    dtype = get_torch_dtype(dtype)
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # apply attention
    if (version is None or version == 3) and flash_attn_3_available():
        import flash_attn_interface  # type: ignore[import-not-found,import-untyped,unused-ignore]

        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    elif flash_attn_2_available():
        import flash_attn  # type: ignore[import-not-found,import-untyped,unused-ignore]

        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        raise RuntimeError("Flash attention is not available.")

    # output
    return x.type(out_dtype)  # type: ignore[no-any-return]


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: str | torch.dtype = "bfloat16",
    version: int | None = None,
) -> torch.Tensor:
    """
    :param q: query tensor, shape [batch, seq_len_q, dim]
    :param k: key tensor, shape [batch, seq_len_k, dim]
    :param v: value tensor, shape [batch, seq_len_k, dim]
    :param q_lens: query sequence lengths, shape [batch]
    :param k_lens: key sequence lengths, shape [batch]
    :param dropout_p: dropout probability
    :param softmax_scale: scale factor for softmax
    :param q_scale: scale factor for query
    :param causal: whether to use causal attention
    :param window_size: window size for local attention
    :param deterministic: whether to use deterministic attention
    :param dtype: data type for computation
    :param fa_version: version of flash attention, 2 or 3
    :return: output tensor, shape [batch, seq_len_q, dim]
    """
    if flash_attn_available():
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,
        )
    else:
        dtype = get_torch_dtype(dtype)
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out

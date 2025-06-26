# Modified from https://github.com/Wan-Video/Wan2.1/wan/modules/clip.py
# First modified from ``https://github.com/openai/CLIP"" and ``https://github.com/mlfoundations/open_clip""
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T  # type: ignore[import-untyped]
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from ..utils import PretrainedMixin, attention, pos_interpolate
from .xlm import XLMRobertaWithHead

__all__ = [
    "XLMRobertaCLIP",
]


class QuickGELU(nn.Module):
    """
    QuickGELU activation function.
    This is a faster approximation of the GELU activation function.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        QuickGELU activation function.
        """
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """
    Layer normalization that supports float16 inputs
    but calculates in float32.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape [B, L, C].
        :return: Normalized tensor of the same shape.
        """
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):
    """
    Self-attention layer with multi-head attention.
    This layer computes the attention scores and applies them to the input.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        causal: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        """
        Self-attention layer.
        :param dim: Input dimension.
        :param num_heads: Number of attention heads.
        :param causal: Whether to use causal attention.
        :param attn_dropout: Dropout probability for attention weights.
        :param proj_dropout: Dropout probability for the output projection.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape [B, L, C].
        :return: Output tensor of shape [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q, k, v = self.to_qkv(x).view(b, s, 3, n, d).unbind(2)

        # compute attention
        p = self.attn_dropout if self.training else 0.0
        x = attention(q, k, v, dropout_p=p, causal=self.causal, version=2)
        x = x.reshape(b, s, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    This is a combination of the Swish activation function and a linear transformation.
    """

    def __init__(self, dim: int, mid_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        # layers
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape [B, L, C].
        :return: Output tensor of shape [B, L, C].
        """
        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attention block with self-attention and MLP.
    This block applies self-attention followed by an MLP, with optional normalization.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        num_heads: int,
        post_norm: bool = False,
        causal: bool = False,
        activation: Literal["quick_gelu", "gelu", "swi_glu"] = "quick_gelu",
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        """
        Attention block with self-attention and MLP.
        :param dim: Input dimension.
        :param mlp_ratio: Ratio of MLP hidden dimension to input dimension.
        :param num_heads: Number of attention heads.
        :param post_norm: Whether to apply normalization after each block.
        :param causal: Whether to use causal attention.
        :param activation: Activation function for MLP.
        :param attn_dropout: Dropout probability for attention weights.
        :param proj_dropout: Dropout probability for the output projection.
        :param norm_eps: Epsilon for layer normalization.
        """
        assert activation in ["quick_gelu", "gelu", "swi_glu"]
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # layers
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim,
            num_heads,
            causal,
            attn_dropout,
            proj_dropout,
        )
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        if activation == "swi_glu":
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(  # type: ignore[assignment]
                nn.Linear(dim, int(dim * mlp_ratio)),
                QuickGELU() if activation == "quick_gelu" else nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
                nn.Dropout(proj_dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape [B, L, C].
        :return: Output tensor of shape [B, L, C].
        """
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Module):
    """
    Attention pooling layer that computes a weighted average of the input
    sequence using self-attention.

    This layer uses a learnable class token and applies self-attention to
    compute the output.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        num_heads: int,
        activation: Literal["quick_gelu", "gelu"] = "quick_gelu",
        proj_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        """
        Attention pooling layer that computes a weighted average of the input
        sequence using self-attention.
        :param dim: Input dimension.
        :param mlp_ratio: Ratio of MLP hidden dimension to input dimension.
        :param num_heads: Number of attention heads.
        :param activation: Activation function for MLP.

        :param proj_dropout: Dropout probability for the output projection.
        :param norm_eps: Epsilon for layer normalization.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        self.norm_eps = norm_eps

        # layers
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == "quick_gelu" else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape [B, L, C].
        :return: Output tensor of shape [B, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.to_q(self.cls_embedding).view(1, 1, n, d).expand(b, -1, -1, -1)
        k, v = self.to_kv(x).view(b, s, 2, n, d).unbind(2)

        # compute attention
        x = attention(q, k, v, version=2)
        x = x.reshape(b, 1, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)

        # mlp
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image feature extraction.

    This model applies a series of transformer blocks to the input image,
    with optional pooling and normalization.
    """

    head: nn.Parameter | nn.Linear | AttentionPool

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        mlp_ratio: int = 4,
        out_dim: int = 512,
        num_heads: int = 12,
        num_layers: int = 12,
        pool_type: Literal["token", "token_fc", "attn_pool"] = "token",
        pre_norm: bool = True,
        post_norm: bool = False,
        activation: Literal["quick_gelu", "gelu"] = "quick_gelu",
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        """
        Vision Transformer for image feature extraction.
        :param image_size: Size of the input image (assumed square).
        :param patch_size: Size of each patch.
        :param dim: Dimension of the model.
        :param mlp_ratio: Ratio of MLP hidden dimension to input dimension.
        :param out_dim: Output dimension of the model.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer layers.
        :param pool_type: Type of pooling
            - "token": Use a learnable class token for pooling.
            - "token_fc": Use a learnable class token and a linear layer for pooling.
            - "attn_pool": Use attention pooling.
        :param pre_norm: Whether to apply layer normalization before the transformer.
        :param post_norm: Whether to apply layer normalization after each block.
        :param activation: Activation function for MLP.
        :param attn_dropout: Dropout probability for attention weights.
        :param proj_dropout: Dropout probability for the output projection.
        :param embedding_dropout: Dropout probability for the input embeddings.
        :param norm_eps: Epsilon for layer normalization.
        """
        super().__init__()
        assert pool_type in ("token", "token_fc", "attn_pool")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim or dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.norm_eps = norm_eps

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=not pre_norm
        )

        if pool_type in ("token", "token_fc"):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))

        self.pos_embedding = nn.Parameter(
            gain
            * torch.randn(
                1,
                self.num_patches + (1 if pool_type in ("token", "token_fc") else 0),
                dim,
            )
        )
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    dim,
                    mlp_ratio,
                    num_heads,
                    post_norm,
                    False,
                    activation,
                    attn_dropout,
                    proj_dropout,
                    norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = LayerNorm(dim, eps=norm_eps) if post_norm else None

        # head
        if pool_type == "token":
            self.head = nn.Parameter(gain * torch.randn(dim, self.out_dim))
        elif pool_type == "token_fc":
            self.head = nn.Linear(dim, self.out_dim)
        elif pool_type == "attn_pool":
            self.head = AttentionPool(
                dim, mlp_ratio, num_heads, activation, proj_dropout, norm_eps
            )

    def forward(
        self, x: torch.Tensor, interpolation: bool = False, use_31_block: bool = False
    ) -> torch.Tensor:
        """
        :param x: Input tensor of shape [B, 3, H, W].
        :param interpolation: Whether to interpolate positional embeddings.
        :param use_31_block: Whether to return the output after the 31st block.
        :return: Output tensor of shape [B, L, C] or [B, C] depending on pooling.
        """
        b = x.size(0)

        # embeddings
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        if self.pool_type in ("token", "token_fc"):
            x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        if interpolation:
            e = pos_interpolate(self.pos_embedding, x.size(1))
        else:
            e = self.pos_embedding

        x = self.dropout(x + e)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer
        if use_31_block:
            x = self.transformer[:-1](x)
            return x
        else:
            x = self.transformer(x)
            return x


class XLMRobertaCLIP(ConfigMixin, PretrainedMixin, ModelMixin):
    @register_to_config
    def __init__(
        self,
        embed_dim: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        vision_dim: int = 1280,
        vision_mlp_ratio: int = 4,
        vision_heads: int = 16,
        vision_layers: int = 32,
        vision_pool: Literal["token", "token_fc", "attn_pool"] = "token",
        vision_pre_norm: bool = True,
        vision_post_norm: bool = False,
        activation: Literal["quick_gelu", "gelu"] = "gelu",
        vocab_size: int = 250002,
        max_text_len: int = 514,
        type_size: int = 1,
        pad_id: int = 1,
        text_dim: int = 1024,
        text_heads: int = 16,
        text_layers: int = 24,
        text_post_norm: bool = True,
        text_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        """
        XLM-RoBERTa + CLIP model.
        :param embed_dim: Dimension of the output embeddings.
        :param image_size: Size of the input image (assumed square).
        :param patch_size: Size of each patch.
        :param vision_dim: Dimension of the vision transformer.
        :param vision_mlp_ratio: Ratio of MLP hidden dimension to vision dimension.
        :param vision_heads: Number of attention heads in the vision transformer.
        :param vision_layers: Number of transformer layers in the vision transformer.
        :param vision_pool: Type of pooling in the vision transformer.
            - "token": Use a learnable class token for pooling.
            - "token_fc": Use a learnable class token and a linear layer for pooling.
            - "attn_pool": Use attention pooling.
        :param vision_pre_norm: Whether to apply layer normalization before the vision transformer.
        :param vision_post_norm: Whether to apply layer normalization after each block in the vision transformer.
        :param activation: Activation function for MLP in the vision transformer.
        :param vocab_size: Vocabulary size for the text model.
        :param max_text_len: Maximum sequence length for the text model.
        :param type_size: Number of token types for the text model.
        :param pad_id: Padding token ID for the text model.
        :param text_dim: Dimension of the text model.
        :param text_heads: Number of attention heads in the text model.
        :param text_layers: Number of transformer layers in the text model.
        :param text_post_norm: Whether to apply layer normalization after each block in the text model.
        :param text_dropout: Dropout probability for the text model.
        :param attn_dropout: Dropout probability for attention weights in the vision transformer.
        :param proj_dropout: Dropout probability for the output projection in the vision transformer.
        :param embedding_dropout: Dropout probability for the input embeddings in the vision transformer.
        :param norm_eps: Epsilon for layer normalization.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_post_norm = text_post_norm
        self.norm_eps = norm_eps

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps,
        )
        self.textual = XLMRobertaWithHead(
            vocab_size=vocab_size,
            max_seq_len=max_text_len,
            type_size=type_size,
            pad_id=pad_id,
            dim=text_dim,
            out_dim=embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            post_norm=text_post_norm,
            dropout=text_dropout,
        )
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))

    def forward(
        self, imgs: torch.Tensor, txt_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param imgs: [B, 3, H, W] of torch.float32.
        :param txt_ids:    [B, L] of torch.long.
        :return: Tuple of visual and textual embeddings.
            - visual: [B, C] of torch.float32.
            - textual: [B, C] of torch.float32.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_ids)
        return xi, xt

    def param_groups(self) -> list[dict[str, list[nn.Parameter] | float]]:
        """
        Get parameter groups for optimizer.
        :return: List of parameter groups.
        """
        groups: list[dict[str, list[nn.Parameter] | float]] = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "norm" in n or n.endswith("bias")
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not ("norm" in n or n.endswith("bias"))
                ]
            },
        ]
        return groups

    @torch.no_grad()
    def infer_videos(
        self,
        videos: list[torch.Tensor],
        mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
    ) -> torch.Tensor:
        """
        :param videos: List of video tensors, each of shape [T, 3, H, W].
        :param mean: Mean values for normalization.
        :param std: Standard deviation values for normalization.
        :return: List of visual embeddings for each video.
        """
        # Preprocess videos to proper size and normalize to [0, 1] range
        size = (self.image_size,) * 2
        videos_tensor = torch.cat(
            [
                F.interpolate(
                    u.transpose(0, 1), size=size, mode="bicubic", align_corners=False
                )
                for u in videos
            ]
        )
        videos_tensor = T.normalize(mean=mean, std=std)(
            videos_tensor.mul_(0.5).add_(0.5)
        )

        # forward
        videos_tensor = self.visual(videos_tensor, use_31_block=True)
        return videos_tensor

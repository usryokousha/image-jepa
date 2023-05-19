from functools import partial
import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from jepa.layers import (
    Mlp,
    SwiGLUFFNFused,
    MemoryEfficientAttention,
    NestedTensorBlock
)

logger = logging.getLogger("jepa")

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class Transformer(nn.Module):
    """Self-attention based encoder with nested tensor support."""
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        act_layer=nn.GELU,
        block_fn=NestedTensorBlock,
        ffn_layer="mlp",
    ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): input embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.init_weights()

    def init_weights(self):
        named_apply(init_weights_vit_timm, self)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return x_norm, x
    
def transformer_small(**kwargs):
    model = Transformer(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemoryEfficientAttention),
        **kwargs,
    )
    return model


def transformer_base(**kwargs):
    model = Transformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemoryEfficientAttention),
        **kwargs,
    )
    return model


def transformer_large(patch_size=16, **kwargs):
    model = Transformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemoryEfficientAttention),
        **kwargs,
    )
    return model
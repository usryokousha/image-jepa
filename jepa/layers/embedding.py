# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from typing import List


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)

def extract_valid_region(x, mask):
    return x.masked_select(mask).reshape(x.shape[0], -1, x.shape[-1])

def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
class VisionEmbedding(PatchEmbed):
    def __init__(
            self, 
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            cls_token: bool = False,
            learnable_position: bool = False,
    ):
        super().__init__(
            img_size, 
            patch_size, 
            in_chans, 
            embed_dim, 
            norm_layer, 
            flatten_embedding=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None
        if learnable_position:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_pos_embed, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(self.num_pos_embed, embed_dim)

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    @property
    def num_pos_embed(self):
        if self.cls_token is not None:
            return self.num_patches + 1
        else:
            return self.num_patches
        
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        if self.cls_token:
            patch_pos_embed = pos_embed[:, 1:]
        else:
            patch_pos_embed = pos_embed

        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)
        
    def forward(self, img: Tensor, context_mask: Tensor = None):
        B, _, H, W = img.shape
        x = super().forward(img)
        x = x + self.interpolate_pos_encoding(x, H, W)
        if context_mask is not None:
            assert context_mask.ndim == 2 and context_mask.shape[1] == self.num_patches, \
                f"Mask shape {context_mask.shape} is not consistent with sequence length {self.num_patches}"
            context_mask = context_mask.unsqueeze(-1)
            x = extract_valid_region(x, context_mask)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.pos_embed[:, 0]
            x = torch.cat((cls_tokens, x), dim=1)
        return x
    
class MaskEmbedding(nn.Module):
    def __init__(
        self,
        mask_size: Union[int, Tuple[int, int]] = 14,
        embed_dim: int = 768,      
    ):
        super().__init__()
        self.mask_size = make_2tuple(mask_size)
        self.max_patches = self.mask_size[0] * self.mask_size[1]

        self.embed_dim = embed_dim
        self.mask_tokens = nn.Parameter(torch.zeros(1, self.max_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, target_mask: Tensor = None):
        assert target_mask.ndim == 2 and target_mask.shape[1] == self.max_patches, \
            f"Mask shape {target_mask.shape} is not consistent with sequence length {self.max_patches}"
        target_mask = target_mask.unsqueeze(-1)
        x = self.mask_tokens + self.pos_embed
        x = extract_valid_region(x, target_mask)
        return x
    
class LatentEmbedding(nn.Module):
    def __init__(
        self,
        latent_size: int = 256,
        embed_dim: int = 768,
        codebook_size: int = 8192,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.latent_embed = nn.Embedding(codebook_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, latent_size, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z: Tensor):
        z_tokens = self.latent_embed(z)
        z_tokens = z_tokens + self.pos_embed
        return z_tokens
    
def _get_embedding_slices(total_seq_len: int, num_modules: int, seq_slice_dict: dict, seq_len_dict: dict):
        # get the start stop indices for the ith embedding
        seq_slice_dict[total_seq_len] = []
        for i in range(num_modules):
            if i == 0:
                start = 0
            else:
                start = sum(seq_len_dict[:i])
            stop = start + seq_len_dict[i]
            seq_slice_dict[total_seq_len].append(torch.range(start, stop))

seq_len_cache = {}
seq_slice_cache = {}
def merge_embedding_outputs(embedding_outputs: List[Tensor]):
    seq_lens = []
    for i, out in enumerate(embedding_outputs):
        assert out.ndim == 3, \
                f"Embedding {i} output should have shape [batch, seq_len, embed_dim], got {out.ndim}"
        seq_lens.append(out.shape[1])

    total_seq_len = sum(seq_lens)
    if total_seq_len not in seq_slice_cache.keys():
        seq_len_cache[total_seq_len] = seq_lens
        _get_embedding_slices(total_seq_len, len(embedding_outputs), seq_slice_cache, seq_len_cache)

    return torch.cat(embedding_outputs, dim=1)

def extract_embedding(sequence: Tensor, module_idx: int):
    seq_len = sequence.shape[1]
    assert seq_len in seq_slice_cache.keys(), \
        f"Sequence length {seq_len} not found in embedding module {module_idx}"
    return sequence[:, seq_slice_cache[seq_len][module_idx].to(sequence.device)]

class PositionEmbedding(nn.Module):
    def __init__(
            self,
            seq_len: int,
            embed_dim: int = 768,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor, mask: Tensor):
        x = x + extract_valid_region(self.pos_embed, mask.flatten(1).unsqueeze(-1))
        return x

class EmbeddingMerge(nn.Module):
    def __init__(self, embed_modules):
        super(EmbeddingMerge, self).__init__()
        self.embed_modules = nn.ModuleList(embed_modules)
        self.seq_len_dict = {}
        self.seq_slice_dict = {}
        self.num_modules = len(embed_modules)

    def forward(self, *inputs):
        outputs = []
        seq_lens = []
        for i, (module, x) in enumerate(zip(self.embed_modules, inputs)):
            out = module(*x)
            assert out.ndim == 3, \
                f"Embedding {i} output should have shape [batch, seq_len, embed_dim], got {out.ndim}"

            seq_lens.append(out.shape(1))
            outputs.append(out)

        total_len = sum(seq_lens)
        if total_len not in self.seq_len_dict.keys():
            self.seq_len_dict[total_len] = seq_lens
            _get_embedding_slices(total_len, self.num_modules, self.seq_slice_dict, self.seq_len_dict)
        
        # Concatenate along the embedding dimension (assuming it's the last dimension)
        output = torch.cat(outputs, dim=-1)

        return output

    def extract_embedding(self, sequence: Tensor, module_idx: int) -> Tensor:
        seq_len = sequence.shape[1]
        assert seq_len in self.seq_len_dict.keys(), \
            f"Sequence length {seq_len} not found in embedding module {module_idx}"
        return sequence[:, self.seq_slice_dict[seq_len][module_idx].to(sequence.device)]


            


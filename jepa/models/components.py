import torch
from torch import Tensor
import dataclasses
from torch import nn
from torch.nn import functional as F


from .transformer import (Transformer, 
                          TransformerConfig, 
                          BaseTransformerConfig, 
                          SmallTransformerConfig, 
                          LargeTransformerConfig)

from jepa.layers import (
    VisionEmbedding,
    LatentEmbedding,
    MaskEmbedding,
    PositionEmbedding,
    extract_valid_region,
    merge_embedding_outputs,
    extract_embedding
)

from typing import List, Union

class ContextEncoder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        signal_embed: VisionEmbedding,
        transformer: Transformer
    ):
        super().__init__()
        self.signal_embed = signal_embed
        self.transformer = transformer
        self.head_dim = head_dim
        self.head = nn.Linear(transformer.embed_dim, head_dim)

    def forward(self, signal: Tensor , context_mask: Tensor, num_splits: int = 1) -> List[Tensor]:
        signal_list = torch.chunk(signal, num_splits, dim=0)
        context_mask_list = torch.chunk(context_mask, num_splits, dim=0)
        x_list = [self.signal_embed(s, m) for s, m in zip(signal_list, context_mask_list)]
        context_list = self.transformer(x_list)
        return context_list, context_mask_list

class TargetEncoder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        signal_embed: VisionEmbedding,
        transformer: Transformer,
    ):
        super().__init__()
        self.signal_embed = signal_embed
        self.transformer = transformer
        self.head_dim = head_dim
        self.head = nn.Linear(transformer.embed_dim, head_dim)
        
    def forward(self, signal: Tensor , target_mask: Tensor, num_targets: int = 4, num_splits: int = 1) -> Union[Tensor, List[Tensor]]:
        target_mask_list = torch.chunk(target_mask, num_targets, dim=0)
        target_mask_list = [torch.chunk(m, num_splits, dim=0) for m in target_mask_list]

        x = self.signal_embed(signal)
        x = self.transformer(x, target_mask)
        x = self.head(x)

        x_list = torch.chunk(x, num_splits, dim=0)

        outputs = []
        for t_i in target_mask_list:
            targets = [extract_valid_region(x, m) for x, m in zip(x_list, t_i)]
            outputs.append(targets)
        return x

class Predictor(nn.Module):
    def __init__(
        self,
        head_dim: int,
        context_embed: PositionEmbedding,
        mask_embed: MaskEmbedding,
        latent_embed: LatentEmbedding,
        transformer: Transformer,
    ):
        super().__init__()
        self.context_embed = context_embed
        self.mask_embed = mask_embed
        self.latent_embed = latent_embed
        self.transformer = transformer
        self.head_dim = head_dim
        self.head = nn.Linear(transformer.embed_dim, head_dim)

    def forward_features_list(self, 
                context_list: List[Tensor], 
                target_mask_list: List[Tensor],
                z: Tensor) -> Tensor:
        
        outputs = []
        for t_i in target_mask_list:
            embeddings_list = [merge_embedding_outputs(c, self.mask_embed(t), z) for c, t in zip(context_list, t_i)]
            predictions = self.transformer(embeddings_list)
            predictions = [extract_embedding(p, 2) for p in predictions]
        outputs.append(predictions)

        return outputs
        
    def forward(self, 
                context_list: List[Tensor], 
                context_mask_list: List[Tensor],
                target_mask: Tensor,
                z: Tensor, 
                num_targets: int = 4) -> Tensor:
        num_nested_tensors = len(context_list)
        target_mask_list = torch.chunk(target_mask, num_targets, dim=0)
        target_mask_list = [torch.chunk(m, num_nested_tensors, dim=0) for m in target_mask_list]

        context_list = [self.context_embed(c, m) for c, m in zip(context_list, context_mask_list)]
        z_embeddings = self.latent_embed(z)

        return self.forward_features_list(context_list, target_mask_list, z_embeddings)

class NestedMSELoss(torch.nn.MSELoss):
    def forward(self, input_list: Tensor, target_list: Tensor) -> Tensor:
        loss = 0
        for input, target in zip(input_list, target_list):
            loss = loss + super().forward(input, target)
        return loss
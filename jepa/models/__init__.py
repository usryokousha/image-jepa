import logging

from . import transformer
from .transformer import Transformer
from .components import (ContextEncoder, TargetEncoder, Predictor)
from ..layers import VisionEmbedding, LatentEmbedding, MaskEmbedding, PositionEmbedding

logger = logging.getLogger("jepa")

def build_ijepa_models(config):
    transformer_kwargs = dict(
        init_values=config.transformer.layerscale,
        ffn_layer=config.transformer.ffn_layer,
        block_chunks=config.transformer.block_chunks,
        qkv_bias=config.transformer.qkv_bias,
        proj_bias=config.transformer.proj_bias,
        ffn_bias=config.transformer.ffn_bias,
    )

    # predictor
    predictor_transformer = transformer.__dict__[config.predictor.transformer.arch](**transformer_kwargs)
    seq_len = (config.context_encoder.img_size // config.context_encoder.patch_size) ** 2
    predictor_pos_embed = PositionEmbedding(
        seq_len=seq_len,
        embed_dim=predictor_transformer.embed_dim,)
    
    predictor_mask_embed = MaskEmbedding(
        seq_len=seq_len,
        embed_dim=predictor_transformer.embed_dim,)

    predictor = Predictor(
        head_dim=config.predictor.head_dim,
        pos_embed=predictor_pos_embed,
        mask_embed=predictor_mask_embed,)

    # context encoder
    context_transformer = transformer.__dict__[config.context_encoder.transformer.arch]
    context_signal_embed = VisionEmbedding(
        img_size=config.context_encoder.img_size,
        patch_size=config.context_encoder.patch_size,
        embed_dim=context_transformer.embed_dim,)
    
    context_encoder = ContextEncoder(
        head_dim=predictor.transformer.embed_dim,
        signal_embed=context_signal_embed,
        transformer=context_transformer(**transformer_kwargs)
    )

    # target encoder
    target_transformer = transformer.__dict__[config.target_encoder.transformer.arch]
    target_signal_embed = VisionEmbedding(
        img_size=config.target_encoder.img_size,
        patch_size=config.target_encoder.patch_size,
        embed_dim=target_transformer.embed_dim,)
    
    target_encoder = TargetEncoder(
        config.predictor.head_dim,
        signal_embed=target_signal_embed,
        transformer=target_transformer(**transformer_kwargs))
    
    logger.info(f"Context Encoder embedding dim: {context_encoder.transformer.embed_dim}")
    logger.info(f"Target Encoder embedding dim: {target_encoder.transformer.embed_dim}")
    logger.info(f"Predictor embedding dim: {predictor.transformer.embed_dim}")
    return context_encoder, target_encoder, predictor
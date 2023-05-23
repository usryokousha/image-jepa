import logging
import torch
from torch import nn

from jepa.models import build_jepa_models
from typing import Optional, Tuple, List, Union

try:
    from xformers.ops import fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
assert XFORMERS_AVAILABLE, "xFormers is required for JEPA training"

logger = logging.getLogger("jepa")
         
class JEPA(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=config.amp) if \
            config.compute_precision.grad_scaler else None
        
        student_model_dict = dict()
        teacher_model_dict = dict()

        context_encoder, target_encoder, predictor = build_jepa_models(config)

        student_model_dict["target_encoder"] = target_encoder
        teacher_model_dict["context_encoder"] = context_encoder
        teacher_model_dict["predictor"] = predictor

        if config.predictor.pretrained_weights:
            ckpt = torch.load(config.predictor.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {config.predictor.pretrained_weights}")
            target_encoder.load_state_dict(ckpt["model"], strict=False)

        self.predictor_out_dim = predictor.head_dim

        logger.info(f"OPTIONS -- VQ Encoder")
        if config.latent_encoder is not None:
            logger.info(f"OPTIONS -- latent encoder: {config.latent_encoder.arch}")
            logger.info(f"OPTIONS -- latent encoder: {config.latent_encoder.codebook_size}")
            logger.info(f"OPTIONS -- latent encoder: {config.latent_encoder.embed_dim}")

            latent_encoder = build_latent_encoder(config)

            teacher_model_dict["latent_encoder"] = latent_encoder

           


import logging
import torch
import torch.nn as nn
from .transformer import Transformer
from einops.layers.torch import Rearrange

logger = logging.getLogger("jepa")

class VQKD(nn.Module):
    def __init__(
            self,
            encoder: Transformer,
            decoder: Transformer,
            config,
    ):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            encoder,
            nn.Sequential(
                nn.Linear(encoder.embed_dim, 4 * encoder.embed_dim),
                nn.Tanh(),
                nn.Linear(4 * encoder.embed_dim, config.latent_encoder.embed_dim),
            )
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, decoder.embed_dim),
            nn.Sequential(
                nn.Linear(decoder.embed_dim, 4 * decoder.embed_dim),
                nn.Tanh(),
                nn.Linear(4 * decoder.embed_dim, config.latent_encoder.kd_dim),
            )

        
    
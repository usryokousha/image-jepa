import logging
import torch
import torch.nn as nn

logger = logging.getLogger("jepa")

class VQKD(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            config,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        
    
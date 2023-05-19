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

from .components import (
    ContextEncoder,
    TargetEncoder,
    Predictor,
    JEPAConfig)

from typing import Optional, Tuple, List, Union
         
class JEPA(nn.Module):
    def __init__(
        self,
        config: JEPAConfig 
    ):
        super().__init__()
        self.config = config
        self.context_encoder = ContextEncoder(config.context_encoder)
        self.target_encoder = TargetEncoder(config.target_encoder)
        self.predictor = Predictor(config.predictor)

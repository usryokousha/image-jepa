import logging
import torch
from torch import nn

from jepa.models import build_jepa_models
from jepa.layers.loss import VICReg
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

        teacher_model_dict["target_encoder"] = target_encoder
        student_model_dict["context_encoder"] = context_encoder
        student_model_dict["predictor"] = predictor

        if config.target_encoder.pretrained_weights:
            ckpt = torch.load(config.target_encoder.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {config.target_encoder.pretrained_weights}")
            target_encoder.load_state_dict(ckpt["model"], strict=False)

        self.predictor_out_dim = predictor.head_dim
        self.target_embed_dim = target_encoder.transformer.embed_dim
        self.context_embed_dim = context_encoder.transformer.embed_dim
        self.predictor_embed_dim = predictor.transformer.embed_dim

        self.do_vicreg = config.vicreg.loss_weight > 0.0

        logger.info(f"OPTIONS -- Latent Encoder")
        if config.latent_encoder is not None:
            logger.info(f"OPTIONS -- latent encoder: {config.latent_encoder.arch}")
            logger.info(f"OPTIONS -- latent encoder: {config.latent_encoder.codebook_size}")
            logger.info(f"OPTIONS -- latent encoder: {config.latent_encoder.embed_dim}")

            # TODO: build latent encoder function
            latent_encoder = build_latent_encoder(config)

            student_model_dict["latent_encoder"] = latent_encoder

            if config.latent_encoder.pretrained_weights:
                ckpt = torch.load(config.latent_encoder.pretrained_weights)
                logger.info(f"OPTIONS -- pretrained weights: loading from {config.latent_encoder.pretrained_weights}")
                latent_encoder.load_state_dict(ckpt["model"], strict=False)

            self.latent_embed_dim = latent_encoder.transformer.embed_dim
            self.do_latent = True

        else:
            self.do_latent = False
            logger.info(f"OPTIONS -- Latent Endoder -- No Latent")
        
        logger.info(f"OPTIONS -- VICReg")
        if self.do_vicreg:
            logger.info(f"OPTIONS -- VICReg -- loss weight: {config.vicreg.loss_weight}")
            logger.info(f"OPTIONS -- VICReg -- embed dim: {config.vicreg.embed_dim}")
            logger.info(f"OPTIONS -- VICReg -- num targets: {config.vicreg.num_features}")

            self.loss = VICReg(
                embed_dim=self.target_embed_dim,
                num_features=self.predictor_out_dim,
                sim_coeff=config.vicreg.sim_coeff,
                std_coeff=config.vicreg.std_coeff,
                cov_coeff=config.vicreg.cov_coeff,
            )
        else:
            logger.info(f"OPTIONS -- VICReg -- Not using VICReg")
            self.loss = nn.MSELoss()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backprogation through the teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {config.target_encoder.transformer.arch}-based networks")

    def forward(self, inputs):
        raise NotImplementedError
    
    def backprop_loss(self, loss):
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def forward_backward(self, images):
        num_splits = self.config.batch_size // self.config.batch_split
        image_crops = images["images"]
        context_masks = images["collated_context_masks"]
        target_masks = images["collated_target_masks"]

        @torch.no_grad()
        def get_teacher_output():
            target_patch_list = self.teacher["target_encoder"](image_crops, target_masks, num_splits=num_splits)
            return target_patch_list
        
        teacher_target_patch_list = get_teacher_output()

        student_context_patch_list = self.student["context_encoder"](image_crops, context_masks, num_splits=num_splits)

        if self.do_latent:
            with torch.no_grad():
                student_latent_patches = self.student["latent_encoder"](image_crops, context_masks)

        student_target_patch_list = self.student["predictor"](
            student_context_patch_list, 
            context_masks, 
            target_masks, 
            z=student_latent_patches, 
            num_splits=num_splits)
        
        _, student_cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(student_target_patch_list)
        _, teacher_cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(teacher_target_patch_list)

        loss = self.loss(student_cat_inputs, teacher_cat_inputs) / self.config.num_targets
        self.backprop_loss(loss)

        return loss
    
    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.teacher.keys():
                student_param_list += list(self.student[k].parameters())
                teacher_param_list += list(self.teacher[k].parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1.0 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    
        


        

        

           


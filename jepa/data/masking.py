import math
import torch
import torch.distributed as dist

from timm.layers import to_2tuple

from typing import Optional, Tuple

def check_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2, f"Expecting tuple of length 2, got {x}"
        return x
    
import math
import numpy as np

def check_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2, f"Expecting tuple of length 2, got {x}"
        return x

class MaskGenerator:
    def __init__(
            self,
            image_size: tuple = (224, 224),
            patch_size: tuple = (16, 16),
            target_aspect_ratio: tuple = (0.75, 1.5),
            target_scale: tuple = (0.15, 0.2),
            context_aspect_ratio: tuple = None,
            context_scale: tuple = (0.85, 1.0),
            num_targets: int = 4
    ):
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        patch_grid_size = tuple(map(lambda x, y: x // y, image_size, patch_size))

        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        target_aspect_ratio = check_2tuple(target_aspect_ratio)
        target_scale = check_2tuple(target_scale)
        context_scale = check_2tuple(context_scale)

        if context_aspect_ratio is not None:
            context_aspect_ratio = check_2tuple(context_aspect_ratio)
            self.context_aspect_ratio = tuple(map(math.log, context_aspect_ratio))

        self.target_aspect_ratio = tuple(map(math.log, context_aspect_ratio))
        self.target_scale = target_scale
        self.context_scale = context_scale
        self.num_targets = num_targets

    def _generate_masks(self, scale, aspect_ratio, num_masks=1):
        area = int(self.num_patches * scale)

        h = int(round(np.sqrt(area / aspect_ratio)))
        w = int(round(np.sqrt(area * aspect_ratio)))

        mask = np.zeros((num_masks, self.patches_resolution[0], self.patches_resolution[1]), dtype=bool)
        for i in range(num_masks):
            top = np.random.randint(0, self.patches_resolution[0] - h)
            left = np.random.randint(0, self.patches_resolution[1] - w)
            mask[i, top:top + h, left:left + w] = 1
        return mask

    def generate(self, context_scale, context_aspect_ratio, target_scale, target_aspect_ratio, num_targets=4):
        target_masks = self._generate_masks(target_aspect_ratio, target_scale, num_targets)
        context_masks = self._generate_masks(context_aspect_ratio, context_scale, 1)
        context_masks = np.logical_and(context_masks, np.logical_not(target_masks.any(axis=0)))
        return context_masks, target_masks

    def get_params(self):
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])
        if self.context_aspect_ratio is None:
            context_aspect_ratio = 1.
        else:
            context_aspect_ratio = np.exp(np.random.uniform(self.context_aspect_ratio[0], self.context_aspect_ratio[1]))

        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        target_aspect_ratio = np.exp(np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1]))
        return context_scale, context_aspect_ratio, target_scale, target_aspect_ratio

    def forward(self,
                context_scale,
                context_aspect_ratio,
                target_scale,
                target_aspect_ratio):

        context_masks, target_masks = self.generate(
            context_scale, 
            context_aspect_ratio, 
            target_scale, 
            target_aspect_ratio, 
            self.num_targets)
            
        return context_masks, np.split(target_masks, self.num_targets, axis=0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(image_size={self.img_size}, patch_size={self.patch_size}, "
            f"target_aspect_ratio={self.target_aspect_ratio}, target_scale={self.target_scale}, "
            f"context_aspect_ratio={self.context_aspect_ratio}, context_scale={self.context_scale}, "
            f"num_targets={self.num_targets}"
        )


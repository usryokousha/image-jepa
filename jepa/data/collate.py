import torch
from .masking import MaskGenerator

def collate_data(
        batch,
        mask_generator: MaskGenerator,
        num_splits: int = 1):
    assert len(batch) % num_splits == 0, "batch size must be divisible by num_splits"

    # special masking collation
    nested_batch_size = batch.shape[0] // num_splits
    context_mask_list = []
    target_mask_list = []
    for _ in range(num_splits):
        params = mask_generator.get_params()
        sub_mask_list = [[]] * mask_generator.num_targets
        for _ in range(nested_batch_size):
            context_mask, target_mask = mask_generator(*params)
            context_mask_list.append(torch.BoolTensor(context_mask))
            for i, m in enumerate(target_mask):
                sub_mask_list[i].append(torch.BoolTensor(m))
            torch.stack(sub_mask_list)
            
    context_masks = torch.stack(context_mask_list)
    target_masks = torch.stack(target_mask_list)
    return torch.stack(batch), context_masks, target_masks
import torch

def generate_box_mask(batch_size, channels, height, width, box_size=16, device='cuda'):
    """
    Generate center box masks for inpainting.
    
    Returns:
        mask: [B, C, H, W], 1 = observed, 0 = masked (center box)
    """
    mask = torch.ones(batch_size, channels, height, width, device=device)
    start = (height - box_size) // 2
    end = start + box_size
    mask[:, :, start:end, start:end] = 0
    return mask

def generate_random_box_mask(batch_size, channels, height, width, min_box_size=13, max_box_size=18, device='cuda'):
    """
    Generate center box masks with RANDOM box sizes for each sample in the batch.
    
    Args:
        batch_size: number of samples
        channels: number of channels
        height, width: image dimensions
        min_box_size: minimum box size (default: 1 pixel)
        max_box_size: maximum box size (default: full image size)
    
    Returns:
        mask: [B, C, H, W], 1 = observed, 0 = masked (center box with random size per sample)
    """
    if max_box_size is None:
        max_box_size = min(height, width)
    
    masks = []
    for _ in range(batch_size):
        # Randomly sample box size for this sample
        box_size = torch.randint(min_box_size, max_box_size + 1, (1,)).item()
        
        # Create mask for this sample
        mask = torch.ones(channels, height, width, device=device)
        start = (height - box_size) // 2
        end = start + box_size
        mask[:, start:end, start:end] = 0
        masks.append(mask)
    
    return torch.stack(masks, dim=0)

def generate_random_mask(batch_size, channels, height, width, mask_ratio=0.5, device='cuda'):
    """
    Generate random pixel masks for inpainting.
    
    Args:
        mask_ratio: proportion of pixels to mask
    
    Returns:
        mask: [B, C, H, W], 1 = observed, 0 = masked
    """
    mask = torch.rand(batch_size, channels, height, width, device=device)
    mask = (mask > mask_ratio).float()
    return mask
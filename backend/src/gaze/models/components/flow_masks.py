import torch


def create_simple_mask(dim: int, alternate_pattern: bool = True):
    """
    Create the simplest binary mask for normalizing flows.

    Parameters:
    -----------
    dim : int
        Total number of dimensions
    alternate_pattern : bool
        If True, alternates mask. If False, splits first half/second half.

    Returns:
    --------
    torch.Tensor
        A 1D binary mask of length dim

    Examples:
        # Alternate pattern (1, 0, 1, 0, ...)
        mask_alternate = create_simple_mask(4)
        # Output: tensor([1, 0, 1, 0])

        # Split pattern (1, 1, 0, 0, ...)
        mask_split = create_simple_mask(4, alternate_pattern=False)
        # Output: tensor([1, 1, 0, 0])
    """
    if alternate_pattern:
        # Create alternating mask (1, 0, 1, 0, ...)
        mask = torch.tensor([1, 0] * ((dim + 1) // 2))[:dim]
    else:
        # Create split mask (1 for first half, 0 for second half)
        mask = torch.zeros(dim, dtype=torch.long)
        mask[: dim // 2] = 1

    return mask

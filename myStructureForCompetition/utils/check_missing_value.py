import torch

def check_nan(x: torch.Tensor) -> bool:
    """ Check if there is NaN in tensor """
    checker = False
    if True in torch.isnan(x):
        checker = True
    return checker

def zero_filtering(x: torch.Tensor) -> torch.Tensor:
    """
    Add eps value for zero embedding, because competition metric is cosine similarity
    Cosine Similarity will be returned NaN, when input value has zero, like as torch.clamp()
    """
    eps = 1e-4
    if (x <= eps).any():
        print(f"Notice: Some values are less than or equal to eps ({eps}). Replacing with eps.")
        x[x <= eps] = eps
    return x

def nan_filtering(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Change eps value for NaN Embedding, because competition metric is cosine similarity
    Cosine Similarity will be returned NaN
    """
    return torch.nan_to_num(x, nan=eps)
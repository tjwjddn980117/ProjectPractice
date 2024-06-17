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
    # x의 복사본을 생성하여 inplace 연산을 피함
    x_filtered = x.clone()
    
    # 조건을 만족하는 경우에만 값을 대체하고 메시지 출력
    if (x_filtered <= eps).any():
        print(f"\n!!!! Notice: Some values are less than or equal to eps ({eps}). Replacing with eps.")
        x_filtered[x_filtered <= eps] = eps
    
    return x_filtered

def nan_filtering(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Change eps value for NaN Embedding, because competition metric is cosine similarity
    Cosine Similarity will be returned NaN
    """
    return torch.nan_to_num(x, nan=eps)
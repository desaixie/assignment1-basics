import torch
import torch.nn as nn
from jaxtyping import Float, Int
import einops

def softmax(x: Float[torch.Tensor, "..."], dim=-1) -> Float[torch.Tensor, "..."]:
    maximum = torch.max(x, dim=dim, keepdim=True).values  # e: remember to add the .values, as the function turns a named tuple (values, indices)
    x = x - maximum
    exp_x = x.exp()
    normalizer = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / normalizer
    
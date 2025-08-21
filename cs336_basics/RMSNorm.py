import torch
import torch.nn as nn
from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Float[torch.Tensor, "batch seqlen d_model"]) -> Float[torch.Tensor, "batch seqlen d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # upcast, to prevent overflow when squaring

        normalizer = torch.sqrt( (x.square().sum(dim=-1, keepdim=True) + self.eps) / self.d_model)  # "batch seqlen d_model"
        x = x / normalizer * self.gain

        x = x.to(in_dtype)
        return x
    
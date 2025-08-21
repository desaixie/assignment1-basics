import torch
import torch.nn as nn
from jaxtyping import Float
from cs336_basics.linear import Linear

def find_multiple_64(d_ffn_raw):
    # first find the one larger than and the one smaller than d_ffn_raw. then pick the one closer
    smaller = larger = 1
    while True:
        if larger >= d_ffn_raw:
            smaller = int(larger // 64)
            break
        else:
            larger *= 64
    if abs(smaller - d_ffn_raw) <= abs(larger - d_ffn_raw):
        return smaller
    return larger

class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        
        d_ffn = find_multiple_64(int(d_model * 8 // 3)) if d_ff is None else d_ff
        print(f"d_ffn: {d_ffn}")
        
        self.linear1 = Linear(d_model, d_ffn, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ffn, device=device, dtype=dtype)
        self.linear2 = Linear(d_ffn, d_model, device=device, dtype=dtype)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Float[torch.Tensor, "batch seqlen d_model"]) -> Float[torch.Tensor, "batch seqlen d_model"]:
        x1 = self.linear1(x)
        gate = self.sigmoid(x1) * x1
        x = gate * self.linear3(x)  # SwiGLU: gate * content, element-wise multiply
        x = self.linear2(x)
        return x
        
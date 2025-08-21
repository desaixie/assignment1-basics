import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None =None, dtype: torch.dtype | None =None):
        super().__init__()  # e: remember calling super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w = torch.empty(out_features, in_features, device=device, dtype=dtype), 
        std = torch.sqrt(2/(in_features, out_features))
        self.W = nn.Parameter( torch.nn.init.trunc_normal_(w, mean=0., std=std, a=-3.*std, b=3.*std))  # initialization
        # e: need to be out, in, following pytorch
        
    def forward(self, x: torch.Tensor):
        return x @ self.W.T  # (..., d_in) x (d_in, d_out) -> (..., d_out)
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        s = {}
        s['W'] = state_dict
        return super().load_state_dict(s, strict, assign)
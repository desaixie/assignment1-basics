import torch
import torch.nn as nn
from jaxtyping import Float

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device|None = None, dtype: torch.dtype | None = None):
        super().__init__()
        emb = nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), mean=0., std=1., a=-3., b=3.)
        self.emb = nn.Parameter(emb)
    
    def forward(self, token_ids: Float[torch.Tensor, "batch seqlen"]) -> Float[torch.Tensor, "batch seqlen d_model"]:
        return self.emb[token_ids]
        
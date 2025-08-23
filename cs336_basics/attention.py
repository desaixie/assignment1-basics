import torch
import torch.nn as nn
from jaxtyping import Float, Int, Bool
import einops

def softmax(x: Float[torch.Tensor, "..."], dim=-1) -> Float[torch.Tensor, "..."]:
    maximum = torch.max(x, dim=dim, keepdim=True).values  # e: remember to add the .values, as the function turns a named tuple (values, indices)
    x = x - maximum
    exp_x = x.exp()
    normalizer = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / normalizer
    
# passed test in one try
def scaled_dot_product_attention(
            Q: Float[torch.Tensor, "batch ... seqlen d_k"], 
            K: Float[torch.Tensor, "batch ... seqlen d_k"], 
            V: Float[torch.Tensor, "batch ... seqlen d_v"],
            mask: Bool[torch.Tensor, "seqlen seqlen"] | None = None,
        ) -> Float[torch.Tensor, "batch ... d_v"]:
    d_k = Q.shape[-1]
    affinity_logits = Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(d_k)) # "batch ... seqlen seqlen"
    
    # mask
    if mask is not None:
        adding_mask = torch.where(mask, torch.zeros_like(mask), torch.ones_like(mask) * float('-inf'))
        affinity_logits += adding_mask
    
    affinity_score = softmax(affinity_logits, dim=-1)  # "batch ...", normalized over num keys dim
    attention_value = affinity_score @ V
    return attention_value
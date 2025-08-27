import torch
import torch.nn as nn
import numpy as np
from jaxtyping import Float, Int
from collections.abc import Callable
from typing import List, Tuple
import math

"""implements a sample without replacement dataset"""
class Dataloader:
    def __init__(self, x: Int[np.ndarray, "data_size"], context_length: int, device: str):
        x = torch.from_numpy(x)
        self.data = x
        self.context_length =context_length

        data_size = x.shape[0]
        valid_indices = torch.arange(data_size - context_length)  # e: should start from 0 not 1, error in handout
        self.starting_indices = valid_indices[torch.randperm(valid_indices.shape[0])]
        self.cur_position = 0
        self.device = device

    def next(self) -> Tuple[Int[torch.Tensor, "contextlen"], Int[torch.Tensor, "contextlen"]]:
        starting_ind = self.starting_indices[self.cur_position]
        self.cur_position += 1

        indices = starting_ind + torch.arange(self.context_length)
        token_ids = self.data[indices].to(device=self.device)
        target_ids = self.data[indices+1].to(device=self.device)
        return token_ids, target_ids

def collate_fn(tuple_list: List[Tuple[Int[torch.Tensor, "contextlen"], Int[torch.Tensor, "contextlen"]]]) -> Tuple[Int[torch.Tensor, "batch contextlen"], Int[torch.Tensor, "batch contextlen"]]:
    l = list(zip(*tuple_list))
    return torch.stack(l[0], dim=0), torch.stack(l[1], dim=0)


""" sample with replacement"""
def dataloader(x: Int[np.ndarray, "data_size"], batch_size: int, context_length: int, device: str) -> Tuple[Int[torch.Tensor, "batch contextlen"], Int[torch.Tensor, "batch contextlen"]]:
    x = torch.from_numpy(x)
    data_size = x.shape[0]
    i = torch.randint(0, data_size - context_length, size=(batch_size,1))  # e: should start from 0 not 1, error in handout
    indices_batch = i + torch.arange(context_length).unsqueeze(0)
    token_ids = x[indices_batch].to(device=device)
    target_ids = x[indices_batch+1].to(device=device)
    return token_ids, target_ids
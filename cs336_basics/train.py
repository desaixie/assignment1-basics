import torch
import torch.nn as nn
import numpy as np
from jaxtyping import Float, Int
from collections.abc import Callable
from typing import List, Tuple
import math
import os
import typing
import argparse
from pathlib import Path
import datetime
from functools import partial

from cs336_basics.transformer import Transformer_LM
from cs336_basics.cross_entropy import cross_entropy_loss
from cs336_basics.perplexity import perplexity
from cs336_basics.AdamW import AdamW, gradient_clipping
from cs336_basics.lr_scheduler import lr_cosine_schedule

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
    
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(ckpt, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]
    
def train(args):
    now = datetime.now()
    # consturct model
    model = Transformer_LM(args.d_model, args.num_heads, args.vocab_size, args.context_length, args.num_layers, args.theta, args.context_length, args.device, args.model_dtype)
    model.to(args.device)

    # construct optimizer
    lr_scheduler = partial(lr_cosine_schedule, lrmax=args.lrmax, lrmin=args.lrmin, num_warmup=args.num_warmup, num_cosine=args.num_cosine)
    optimizer = AdamW(model.parameters(), 0.1, (args.beta1, args.beta2), args.eps, args.weight_decay)
    
    
    start_step = 0
    if args.resume_path:
        start_step = load_checkpoint(args.resume_path, model, optimizer)
    
    optimizer.param_groups[0]['lr'] = lr_scheduler(start_step)  # update lr in AdamW
        
    # load data
    data = np.memmap(args.data_path)
    dataloader = Dataloader(data, args.context_length, args.device)

    # training loop
    for i in range(start_step, args.train_steps):
        # train_one_step
        tuple_list = [dataloader.next() for _ in range(args.batch_size)]
        batch = collate_fn(tuple_list)
        input_batch, target_batch = batch
        
        output = model(input_batch)
        loss = cross_entropy_loss(output, target_batch)
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm=args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # update lr in AdamW
        lr = lr_scheduler(i)
        for group in optimizer.param_groups:
            group['lr'] = lr
        
        # validation_step
        # TODO separate val data?
        if i % args.val_every_step == 0:
            with torch.no_grad():
                output = model(input_batch)
                losses = cross_entropy_loss(output, target_batch)
                val_loss = perplexity(losses)

        # logging
        if i % args.log_every_step == 0:
            print(f"step: {i}, loss: {loss.item()}, val_loss: {val_loss.item()}")
            # TODO ema of lsses

        # save checkpoint
        if i % args.save_every_step == 0:
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            save_dir = Path(args.save_dir) / timestamp
            save_checkpoint(model, optimizer, i, save_dir / f"step_{i}.ckpt")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--d_model', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--context_length', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--theta', type=float)
    parser.add_argument('--context_length', type=int)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--model_dtype', type=str, default='bf16')

    # optimizer
    parser.add_argument('--lrmax', type=float)
    parser.add_argument('--lrmin', type=float)
    parser.add_argument('--beta1', type=float)
    parser.add_argument('--beta2', type=float)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_warmup', type=int)
    parser.add_argument('--num_cosine', type=int)
    parser.add_argument('--max_grad_norm', type=float)

    # data
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int)
    
    # training loop
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--val_every_step', type=int, default=10)
    parser.add_argument('--log_every_step', type=int, default=10)
    parser.add_argument('--save_every_step', type=int, default=50)
    parser.add_argument('--resume_path', type=str, default='')

    

    args = parser.parse_args()
    if args.model_dtype == 'bf16':
        args.model_dtype = torch.bfloat16
    args.device = torch.device(args.device)
    train(args)
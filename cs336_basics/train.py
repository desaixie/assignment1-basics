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
from datetime import datetime
from functools import partial

from cs336_basics.transformer import Transformer_LM
from cs336_basics.cross_entropy import cross_entropy_loss
from cs336_basics.perplexity import perplexity
from cs336_basics.AdamW import AdamW, gradient_clipping
from cs336_basics.lr_scheduler import lr_cosine_schedule
from cs336_basics.simple_tokenizer import Tokenizer

def load_data_memmap(file_path, dtype=np.int32):
    """Load data using memory mapping for efficient memory usage."""
    print(f"Loading data from {file_path} using memory mapping...")
    return np.memmap(file_path, dtype=dtype, mode='r')

def load_data_regular(file_path, dtype=np.int32):
    """Load data into regular memory."""
    print(f"Loading data from {file_path} into regular memory...")
    data = np.load(file_path)
    print(f"Loaded {len(data)} tokens")
    return data

"""implements a sample without replacement dataset"""
class Dataloader:
    def __init__(self, x: Int[np.ndarray, "data_size"], context_length: int, device: str):
        x = torch.from_numpy(x)
        self.data = x
        self.context_length = context_length

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
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    print(f"time: {timestamp}")
    print(f'args: {args}')

    # tokenizer
    # from https://github.com/kkaitlyn111/cs336-assignment1/blob/main/cs336_basics/training_loop.py
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=args.special_tokens)
    s = "Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, “no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players.” With a few clicks here and there, you can findout that Noah Syndergaard’s fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game’s highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball’s culture: the analytical hobbyist."
    ids = tokenizer.encode(s)
    print(tokenizer.decode(ids))
    
    if args.reuse_pretokens and os.path.exists(args.pretokens_train_path):
        print(f"Reusing existing pretokenized training data from: {args.pretokens_train_path}")
    else:
        print(f"Creating fresh pretokenized training data...")
        tokenizer.pretokenize_file(
            args.train_path,
            args.pretokens_train_path,
            use_parallel=args.use_parallel_pretokenize
        )
        print(f"Saved fresh pretokenized training data to: {args.pretokens_train_path}")

    if args.reuse_pretokens and os.path.exists(args.pretokens_valid_path):
        print(f"Reusing existing pretokenized validation data from: {args.pretokens_valid_path}")
    else:
        print(f"Creating fresh pretokenized validation data...")
        tokenizer.pretokenize_file(
            args.valid_path,
            args.pretokens_valid_path,
            use_parallel=args.use_parallel_pretokenize
        )
        print(f"Saved fresh pretokenized validation data to: {args.pretokens_valid_path}")
    
    # load data based on the specified method
    if not args.use_memmap:
        print("Loading data into regular memory...")
        train_data = load_data_regular(args.pretokens_train_path)
        valid_data = load_data_regular(args.pretokens_valid_path)
    else:
        print("Loading data using memory mapping...")
        train_data = load_data_memmap(args.pretokens_train_path)
        valid_data = load_data_memmap(args.pretokens_valid_path)

    # consturct model
    model = Transformer_LM(args.d_model, args.num_heads, args.d_ff, args.vocab_size, args.context_length, args.num_layers, args.theta, args.context_length, args.device, args.model_dtype)
    model.to(args.device)

    # construct optimizer
    lr_scheduler = partial(lr_cosine_schedule, lrmax=args.lrmax, lrmin=args.lrmin, num_warmup=args.num_warmup, num_cosine=args.num_cosine)
    optimizer = AdamW(model.parameters(), 0.1, (args.beta1, args.beta2), args.eps, args.weight_decay)
    
    
    start_step = 0
    if args.resume_path:
        start_step = load_checkpoint(args.resume_path, model, optimizer)
    
    optimizer.param_groups[0]['lr'] = lr_scheduler(start_step)  # update lr in AdamW
        
    # load data
    # data = np.memmap(args.data_path)
    dataloader = Dataloader(train_data, args.context_length, args.device)
    val_dataloader = Dataloader(valid_data, args.context_length, args.device)

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
                tuple_list = [val_dataloader.next() for _ in range(args.batch_size)]
                batch = collate_fn(tuple_list)
                input_batch, target_batch = batch
                output = model(input_batch)
                losses = cross_entropy_loss(output, target_batch)
                val_loss = perplexity(losses)

        # logging
        if i % args.log_every_step == 0:
            print(f"step: {i}, loss: {loss.item()}, val_loss: {val_loss.item()}")
            # TODO ema of lsses

        # save checkpoint
        if i % args.save_every_step == 0:
            save_dir = Path(args.save_dir) / timestamp
            save_checkpoint(model, optimizer, i, save_dir / f"step_{i}.ckpt")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--theta', type=float, default=10000.)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--model_dtype', type=str, default='bf16')

    # optimizer
    parser.add_argument('--lrmax', type=float, default=1e-4)
    parser.add_argument('--lrmin', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup', type=int, default=0)
    parser.add_argument('--num_cosine', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # tokenizer
    parser.add_argument("--train_path", type=str, default = "data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid_path", type=str, default = "data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_path", type=str, default = "tinystories_vocab.pkl")
    parser.add_argument("--merges_path", type=str, default = "tinystories_merges.pkl")
    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--pretokens_train_path", type=str, default="data/openweb-train-tokenized.npy", help="Path to pretokenized training data")
    parser.add_argument("--pretokens_valid_path", type=str, default="data/openweb-valid-tokenized.npy", help="Path to pretokenized validation data")
    parser.add_argument("--reuse_pretokens", action="store_true", default = True, help="Reuse existing pretokenized data if available")
    parser.add_argument("--use_memmap", type=bool, default=False)
    parser.add_argument("--use_parallel_pretokenize", type=bool, default=True)  # Default to parallel for full dataset

    # data
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=128)  # batch_size * train_steps * context_length = 327680000
    
    # training loop
    parser.add_argument('--train_steps', type=int, default=1e4)
    parser.add_argument('--val_every_step', type=int, default=10)
    parser.add_argument('--log_every_step', type=int, default=10)
    parser.add_argument('--save_every_step', type=int, default=50)
    parser.add_argument('--resume_path', type=str, default='')

    

    args = parser.parse_args()
    if args.model_dtype == 'bf16':
        args.model_dtype = torch.bfloat16
    args.device = torch.device(args.device)
    train(args)
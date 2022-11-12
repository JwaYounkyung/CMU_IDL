import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import torch.utils.data as torchdata
from torch.utils.data import DataLoader

from dataloader import CustomDataset

from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
args.distributed = False

if 'WORLD_SIZE' in os.environ:
    gpu_no = int(os.environ['WORLD_SIZE'])
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
else:
    gpu_no = 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

data_loaded = CustomDataset()

train_sampler = None
if args.distributed:
    train_sampler = torchdata.distributed.DistributedSampler(data_loaded)

train_loader = DataLoader(dataset=data_loaded,
                          collate_fn=data_loaded.custom_collate_fn,
                          batch_size=batch_size,
                          shuffle=(train_sampler is None),
                          sampler=train_sampler,
                          pin_memory=True)

model = Model()

if args.distributed:
    model = DDP(model, device_ids=[args.local_rank])
'''Single node, multi-GPUs training.'''
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import argparse
import torchvision
import torchvision.transforms as transforms
from torch.backends import cudnn

from models import *
from utils import progress_bar
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Distributed Training')
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
args = parser.parse_args()


# Init process group
def setup(rank, args):
    print("Initialize Process Group...")
    dist.init_process_group(backend='nccl',
                            init_method='tcp://localhost:1111',
                            world_size=args.world_size,
                            rank=rank)


def cleanup():
    dist.destroy_process_group()


def init_seeds(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def basic(gpu, args):
    rank = args.nr * args.gpus + gpu
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, args)

    # create model and move it to GPU with id rank
    torch.cuda.set_device(gpu)
    # set different seed for different process
    init_seeds(1 + rank)
    # Init Model
    model = ResNet101().cuda(gpu)
    ddp_model = DDP(model, device_ids=[gpu], output_device=gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(ddp_model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-4)
    trainSampler, trainLoader, testLoader = prepare_data(rank, args)
    localtime = time.asctime(time.localtime(time.time()))
    if not os.path.exists('output.csv') and rank == 0:
        with open('output.csv', 'w') as f:
            f.write("Time, Epoch, Accuracy,")
            f.write("\n")
            f.write(localtime + ",")
            f.write(" 0, 0,\n")
    for epoch in range(200):
        trainSampler.set_epoch(epoch)
        train(ddp_model, epoch, criterion, optimizer, trainLoader, gpu)
        test(ddp_model, epoch, testLoader, gpu, rank)
    cleanup()


def run():
    args.world_size = args.gpus * args.nodes
    mp.spawn(basic,
             args=(args,),
             nprocs=args.gpus)


# Data
def prepare_data(rank, args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=args.world_size,
        rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=(train_sampler is None), num_workers=2,
        pin_memory=True, sampler=train_sampler)

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)
    return train_sampler, train_loader, test_loader


def train(net, epoch, criterion, optimizer, trainLoader, gpu):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainLoader, 0):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if gpu == 0:
            progress_bar(batch_idx, len(trainLoader), 'Loss: %.3f'
                         % (train_loss / (batch_idx + 1)))


def test(net, epoch, testLoader, gpu, rank):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if gpu == 0:
                progress_bar(batch_idx, len(testLoader), 'Acc: %.3f%% (%d/%d)'
                             % (100. * correct / total, correct, total))
        # write result
        localtime = time.asctime(time.localtime(time.time()))
        if rank == 0:
            with open('output.csv', 'a') as f:
                f.write(localtime + ",")
                f.write(str(epoch) + ",")
                f.write(str(100. * correct / total) + ",")
                f.write("\n")


if __name__ == '__main__':
    if os.path.exists('output.csv'):
        print("Old output.csv Exists! Delete Or Move It, And Try Again!")
        sys.exit(0)
    run()

'''Train CIFAR10 with PyTorch.'''
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import csv

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    memory_allocated_list, memory_reserved_list, memory_inactive_list = [], [], []
    # print(torch.cuda.memory_stats()["allocated_bytes.all.current"]/1024/1024)
    # print(torch.cuda.memory_stats()["reserved_bytes.all.current"]/1024/1024)
    # print(torch.cuda.memory_stats()["inactive_split_bytes.all.current"]/1024/1024)
    memory_allocated_list.append(torch.cuda.memory_stats()["allocated_bytes.all.current"] / 1024 / 1024)
    memory_reserved_list.append(torch.cuda.memory_stats()["reserved_bytes.all.current"] / 1024 / 1024)
    memory_inactive_list.append(torch.cuda.memory_stats()["inactive_split_bytes.all.current"] / 1024 / 1024)
    check_memory_stat_consistency()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx % 10 == 0:
            memory_allocated_list.append(torch.cuda.memory_stats()["allocated_bytes.all.current"] / 1024 / 1024)
            memory_reserved_list.append(torch.cuda.memory_stats()["reserved_bytes.all.current"] / 1024 / 1024)
            memory_inactive_list.append(torch.cuda.memory_stats()["inactive_split_bytes.all.current"] / 1024 / 1024)
            check_memory_stat_consistency()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if batch_idx % 10 == 0:
            memory_allocated_list.append(torch.cuda.memory_stats()["allocated_bytes.all.current"] / 1024 / 1024)
            memory_reserved_list.append(torch.cuda.memory_stats()["reserved_bytes.all.current"] / 1024 / 1024)
            memory_inactive_list.append(torch.cuda.memory_stats()["inactive_split_bytes.all.current"] / 1024 / 1024)
            check_memory_stat_consistency()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    with open('test.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(memory_allocated_list)
        write.writerow(memory_reserved_list)
        write.writerow(memory_inactive_list)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def check_memory_stat_consistency():
    snapshot = torch.cuda.memory_snapshot()

    expected_each_device = collections.defaultdict(lambda: collections.defaultdict(int))

    # for segment in snapshot:
    for idx, segment in enumerate(snapshot):
        expected = expected_each_device[segment["device"]]
        pool_str = segment["segment_type"] + "_pool"

        # expected["segment.all.current"] += 1
        # expected["segment." + pool_str + ".current"] += 1

        # expected["allocated_bytes.all.current"] += segment["allocated_size"]
        # expected["allocated_bytes." + pool_str + ".current"] += segment["allocated_size"]

        expected["reserved_bytes.all.current"] += segment["total_size"]
        expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

        expected["active_bytes.all.current"] += segment["active_size"]
        expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

        is_split = len(segment["blocks"]) > 1
        real = 0.0
        for block in segment["blocks"]:
            # if block["state"] == "active_allocated":
            #     expected["allocation.all.current"] += 1
            #     expected["allocation." + pool_str + ".current"] += 1
            # if idx == 7 or idx == 8:
            # print("The block size is:", block["size"] / 1024 / 1024, "MB", ", the block state is:", block["state"])

            if block["state"].startswith("active_"):
                expected["active.all.current"] += 1
                expected["active." + pool_str + ".current"] += 1

            if block["state"] == "inactive" and is_split:
                expected["inactive_split.all.current"] += 1
                expected["inactive_split." + pool_str + ".current"] += 1
                expected["inactive_split_bytes.all.current"] += block["size"]
                expected["inactive_split_bytes." + pool_str + ".current"] += block["size"]
                real += block["size"]
        # print("Segment Index:", idx, ", Inactivate split rate: ",
        #       round((real / segment["total_size"] * 100), 2),
        #       "%,   Total Size:",
        #       segment["total_size"] / 1024 / 1024, "MB",
        #       ",   Inactive Size:",
        #       round(real / 1024 / 1024, 2), "MB")
    # print()


for epoch in range(start_epoch, start_epoch + 1):
    train(epoch)
    test(epoch)
    scheduler.step()

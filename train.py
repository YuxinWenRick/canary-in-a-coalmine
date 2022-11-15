# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from models.utils import load_model
from utils import progress_bar, set_random_seed, get_dataset
from randomaug import RandAugment


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default='sgd')
parser.add_argument('--resume_checkpoint', '-r', default=None, help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='res18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--bs', default=512, type=int)
parser.add_argument('--size', default=32, type=int)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--num_total', default=None, type=int)
parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
parser.add_argument('--dimhead', default=512, type=int)
parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")
parser.add_argument('--name', default='test')
parser.add_argument('--num_shadow', default=None, type=int)
parser.add_argument('--shadow_id', default=None, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pkeep', default=0.5, type=float)

args = parser.parse_args()

if args.num_shadow is not None:	
    args.job_name = args.name + f'_shadow_{args.shadow_id}'	
else:	
    args.job_name = args.name + '_target'

# take in args
usewandb = not args.nowandb
name = args.job_name
if usewandb:
    import wandb
    wandb.init(project='canary_shadow_model', name=name)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.aug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

tv_dataset = get_dataset(args)


if args.dataset == 'mnist':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

# Add RandAugment with N, M(hyperparameter)
if aug:
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = tv_dataset(root='./data', train=True, download=True, transform=transform_train)
dataset_size = len(trainset)

if args.num_total:
    dataset_size = args.num_total

# set random seed
set_random_seed(args.seed)

# get shadow dataset
if args.num_shadow is not None:
    # get shadow dataset
    keep = np.random.uniform(0, 1, size=(args.num_shadow, dataset_size))
    order = keep.argsort(0)
    keep = order < int(args.pkeep * args.num_shadow)
    keep = np.array(keep[args.shadow_id], dtype=bool)
    keep = keep.nonzero()[0]
else:
    # get target dataset
    keep = np.random.choice(dataset_size, size=int(args.pkeep * dataset_size), replace=False)
    keep.sort()

keep_bool = np.full((dataset_size), False)
keep_bool[keep] = True

trainset = torch.utils.data.Subset(trainset, keep)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)

testset = tv_dataset(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Model factory..
print('==> Building model..')
net = load_model(args)

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
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
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    
    os.makedirs('loglog', exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    with open(f'loglog/{name}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)
    
net.cuda()

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step() # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, 'val_acc': acc, 'lr': optimizer.param_groups[0]['lr'],
        'epoch_time': time.time()-start})

    # Write out csv..
    with open(f'loglog/{name}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 

state = {"model": net.state_dict(),
        "in_data": keep,
        "keep_bool": keep_bool,
        "model_arch": args.net}
os.makedirs('saved_models/' + args.name, exist_ok=True)
torch.save(state, './saved_models/' + args.name + '/' + args.job_name + '_last.pth')

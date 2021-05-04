#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from scipy import spatial
import numpy as np

import moco.loader
import moco.builder
from moco.customDataloader import getLoader as dataLoader
import pdb
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist
import os

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataTrain', metavar='DIR',
                    help='path to train dataset')
parser.add_argument('--dataVal', metavar='DIR',
                    help='path to validation dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default= 'tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--save_path', default='/datad/nkanama/checkpoints', type=str,
                    help='where to save checkpoints')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='use pretrained backbone on imagenet')
parser.add_argument('--name', type=str, required=True, default=False,
                    help='name of model and tensorboard log')

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    #explicit rules
    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()
    main_worker(int(args.gpu), ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):

    writer = SummaryWriter('runs/' + args.name, max_queue=10, flush_secs=60)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    #pdb.set_trace()
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(torch.cuda.device_count())
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    #pdb.set_trace()
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args)
    print(model)
 
    if args.distributed:
        # DistributedDataParallel will use all available devices.
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    #optional resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    traindir  = args.dataTrain
    valDir = args.dataVal
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.Resize((256,256)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.Resize((256,256)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    if args.aug_plus == False:
        augmentation = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    traintransformations = augmentation
    train_dataset = dataLoader(traindir, traintransformations)
    valtransformations = augmentation
    val_dataset = dataLoader(valDir, valtransformations)
    train_sampler = None
    if args.distributed and ngpus_per_node > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    #train and validation for x epochs
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed and ngpus_per_node > 1:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        validation(val_loader, model, criterion, optimizer, epoch, args, writer)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=str(os.path.join(args.save_path,args.name+'checkpoint_{:04d}.pth.tar'.format(epoch))))
    
    #wrap up
    writer.export_scalars_to_json("./" + args.name + "_scalars.json")
    dist.destroy_process_group()
    print(f"{rank} destroy complete")
    writer.close()

'''
function should perform validation test and change hyperparameters after each epoch
set torch.no_grad() and torch.eval()
'''
def validation(val_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    #configure model
    model.eval()
    with torch.no_grad():
        #pass data into model
        for i, (image1, image2) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            if args.gpu is not None:
                image1 = image1.cuda(non_blocking=True)
                image2 = image2.cuda(non_blocking=True)

            # compute output
            output, target, q, k = model(im_q=image1, im_k=image2)
            #similarityMatrix = visualize(q, k, image1, image2, epoch, args, writer)
            loss = criterion(output, target)

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), image1.size(0))
            top1.update(acc1[0], image1.size(0))
            top5.update(acc5[0], image1.size(0))

            #output loss and accuracy top1 to tensorboard
            writer.add_scalar("val loss", loss.item(), epoch * len(val_loader) + i)
            writer.add_scalar("val accuracy", acc1[0], epoch * len(val_loader) + i)
            output_text = " epoch " + str(epoch) + " batch " + str(i) + " accuracy " + str(acc1[0]) + " loss " + str(loss.item())
            writer.add_text("validation stats", output_text, i)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print("validation current epoch results")
                progress.display(i)



def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (image1, image2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            image1 = image1.cuda(non_blocking=True)
            image2 = image2.cuda(non_blocking=True)

        # compute output
        output, target, q, k = model(im_q=image1, im_k=image2)
        #similarityMatrix = visualize(q, k, image1, image2, epoch, args, writer)
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), image1.size(0))
        top1.update(acc1[0], image1.size(0))
        top5.update(acc5[0], image1.size(0))

        #output loss and accuracy top1 to tensorboard
        #pdb.set_trace()
        writer.add_scalar("training loss", loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar("train accuracy", acc1[0], epoch * len(train_loader) + i)
        output_text = " epoch " + str(epoch) + " batch " + str(i) + " accuracy " + str(acc1[0]) + " loss " + str(loss.item())
        writer.add_text("training stats", output_text, i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print("train current epoch results")
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        # 5 rows and 128 columns with prediction values ranging from 0 and up
        pred = pred.t()
        # 5 rows and 128 columns with True or False if predicition matches ground truth (0)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def visualize(a, b, image1, image2, epoch, args, writer):
    # Given that cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
    #                          = dot(u / norm(u), v / norm(v))
    # We fist normalize the rows, before computing their dot products via transposition:
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    #print(res)
   
    # Let's verify with numpy/scipy if our computations are correct:
    a_n = a.detach().cpu().numpy()
    b_n = b.detach().cpu().numpy()
    res_n = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            # cos_sim(u, v) = 1 - cos_dist(u, v)
            res_n[i, j] = 1 - spatial.distance.cosine(a_n[i], b_n[j])
    #print(res_n)

    #write cosine similaity [-1,1] to tensorboard
    for x in range(len(image1)):
        for z in range(len(image2)):
            images = torch.cat((image1[x], image2[z]), 1)
            img_grid = torchvision.utils.make_grid(images, normalize=True, scale_each=True)
            if(x == z and res_n[x][z] <= 0.5):
                output_string = str(epoch) + ' postive sample similarity ' + str(res_n[x][z])
                writer.add_image(output_string, img_grid)
            elif(x != z and res_n[x][z] >= 0.5):
                output_string = str(epoch) + ' negative sample similarity ' + str(res_n[x][z])
                writer.add_image(output_string, img_grid)
    return res_n


if __name__ == '__main__':
    main()

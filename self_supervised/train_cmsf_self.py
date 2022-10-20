import builtins
import os
import sys
import time
import argparse
import pdb
from os.path import join
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from util import adjust_learning_rate, AverageMeter, subset_classes
import models.resnet as resnet
from models.mlp_arch import get_mlp
from tools import get_logger
from util import get_shuffle_ids
from data_loader import get_train_loader


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('data', type=str, help='path to dataset')
    parser.add_argument('--base-dir', default='./',
                        help='projects base directory, different for vision and ada servers')
    parser.add_argument('--exp', default='temp',
                        help='experiment root directory')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'imagenet100'],
                        help='use full or subset of the dataset')
    parser.add_argument('--debug', action='store_true', help='whether in debug mode or not')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=24, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='90,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--cos', action='store_true',
                        help='whether to cosine learning rate or not')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum')

    # model definition
    parser.add_argument('--arch', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18', 'resnet50', 'mobilenet'])

    # Mean Shift
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--mem_bank_size', type=int, default=128000)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--topkp', type=int, default=10)
    parser.add_argument('--weak_strong', action='store_true',
                        help='whether to strong/strong or weak/strong augmentation')

    parser.add_argument('--weights', type=str, help='weights to initialize the model from')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


class ConstrainedMeanShiftSelf(nn.Module):
    def __init__(self, arch, m=0.99, mem_bank_size=128000, topk=5, dataset_size=100, topkp=10):
        super(ConstrainedMeanShiftSelf, self).__init__()

        # save parameters
        self.m = m
        self.mem_bank_size = mem_bank_size
        self.topk = topk
        self.dataset_size = dataset_size
        self.topkp = topkp

        # create encoders and projection layers
        # both encoders should have same arch
        if 'resnet' in arch:
            self.encoder_q = resnet.__dict__[arch]()
            self.encoder_t = resnet.__dict__[arch]()

        # save output embedding dimensions
        # assuming that both encoders have same dim
        feat_dim = self.encoder_q.fc.in_features
        hidden_dim = feat_dim * 2
        proj_dim = feat_dim // 4

        # projection layers
        self.encoder_t.fc = get_mlp(feat_dim, hidden_dim, proj_dim)
        self.encoder_q.fc = get_mlp(feat_dim, hidden_dim, proj_dim)

        # prediction layer
        self.predict_q = get_mlp(proj_dim, hidden_dim, proj_dim)

        # copy query encoder weights to target encoder
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False

        print("using mem-bank size {}".format(self.mem_bank_size))
        print("# Queues {}".format(2))
        # setup queue (For Storing Random Targets)
        self.register_buffer('queue', torch.randn(self.mem_bank_size, proj_dim))
        self.register_buffer('pool', torch.randn(2, self.dataset_size, proj_dim))
        self.register_buffer('pool_qindex', torch.zeros(self.dataset_size).long())
        # normalize the queue embeddings
        self.queue = nn.functional.normalize(self.queue, dim=1)
        # initialize the labels queue (For Purity measurement)
        self.register_buffer('labels', -1*torch.ones(self.mem_bank_size).long())
        self.register_buffer('index_queue', -1 * torch.ones(self.mem_bank_size).long())
        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def data_parallel(self):
        self.encoder_q = torch.nn.DataParallel(self.encoder_q)
        self.encoder_t = torch.nn.DataParallel(self.encoder_t)
        self.predict_q = torch.nn.DataParallel(self.predict_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, targets, labels, indices):
        batch_size = targets.shape[0]

        ptr = int(self.queue_ptr)
        assert self.mem_bank_size % batch_size == 0 

        # replace the targets at ptr (dequeue and enqueue)
        self.pool[self.pool_qindex[indices], indices, :] = targets
        self.pool_qindex[indices] = (self.pool_qindex[indices] + 1) % 2
        self.queue[ptr:ptr + batch_size] = targets
        self.labels[ptr:ptr + batch_size] = labels
        self.index_queue[ptr:ptr + batch_size] = indices
        ptr = (ptr + batch_size) % self.mem_bank_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_t, labels, indices):
        # compute query features
        feat_q = self.encoder_q(im_q)
        # compute predictions for instance level regression loss
        query = self.predict_q(feat_q)
        query = nn.functional.normalize(query, dim=1)

        # compute target features
        with torch.no_grad():
            # update the target encoder
            self._momentum_update_target_encoder()

            # shuffle targets
            shuffle_ids, reverse_ids = get_shuffle_ids(im_t.shape[0])
            im_t = im_t[shuffle_ids]

            # forward through the target encoder
            current_target = self.encoder_t(im_t)
            current_target = nn.functional.normalize(current_target, dim=1)

            # undo shuffle
            current_target = current_target[reverse_ids].detach()
            # update the memory bank
            self._dequeue_and_enqueue(current_target, labels, indices)

        targets = self.queue.clone().detach()

        # calculate distances between vectors
        dist_t = 2 - 2 * torch.einsum('bc,kc->bk', [current_target, targets])
        dist_q = 2 - 2 * torch.einsum('bc,kc->bk', [query, targets])

        # select the k [topk] nearest neighbors [with smallest distance (largest=False)] based on current target
        _, unconstrained_nn_index = dist_t.topk(self.topk, dim=1, largest=False)
        nn_dist_q_unconstrained = torch.gather(dist_q, 1, unconstrained_nn_index)

        # get the previous augmentation for target and memory bank samples
        targets_prime = self.pool[self.pool_qindex[self.index_queue], self.index_queue, :]
        current_target_prime = self.pool[self.pool_qindex[indices], indices, :]

        # select the k' [topkp] nearest neighbors in previous memory bank
        dist_t_prime = 2 - 2 * torch.einsum('bc,kc->bk', [current_target_prime, targets_prime])
        _, nn_index_prime = dist_t_prime.topk(self.topkp, dim=1, largest=False)

        # constrain the primary memory bank search space to NN samples of the previous memory bank
        dist_prime = torch.zeros_like(dist_t)
        dist_prime.scatter_(1, nn_index_prime, -5*torch.ones_like(dist_t))
        dist_t += dist_prime
        _, nn_index_constrained = dist_t.topk(self.topk, dim=1, largest=False)

        # calculate mean shift regression loss
        nn_dist_q_constrained = torch.gather(dist_q, 1, nn_index_constrained)
        loss = ((nn_dist_q_constrained.sum(dim=1) / self.topk).mean()
                + (nn_dist_q_unconstrained.sum(dim=1) / self.topk).mean()) / 2.0

        # purity based on first queue
        labels = labels.unsqueeze(1).expand(nn_dist_q_unconstrained.shape[0], self.topk)
        labels_queue = self.labels.clone().detach()
        labels_queue = labels_queue.unsqueeze(0).expand((nn_dist_q_unconstrained.shape[0], self.mem_bank_size))
        labels_queue = torch.gather(labels_queue, dim=1, index=unconstrained_nn_index)
        matches = (labels_queue == labels).float()
        purity = (matches.sum(dim=1) / self.topk).mean()

        return loss, purity


def main():
    args = parse_option()

    save_dir = join(args.base_dir, 'exp/self_sup_cmsf')
    args.ckpt_dir = join(save_dir, args.exp, 'checkpoints')
    args.logs_dir = join(save_dir, args.exp, 'logs')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args_file = join(args.logs_dir, 'train_args.json')
    s = '*' * 50
    with open(args_file, 'a') as f:
        json.dump(s, f)
        json.dump(vars(args), f, indent=4)

    if not args.debug:
        os.environ['PYTHONBREAKPOINT'] = '0'
        logger = get_logger(
            logpath=os.path.join(args.logs_dir, 'logs'),
            filepath=os.path.abspath(__file__)
        )

        def print_pass(*arg):
            logger.info(*arg)
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print(args)

    train_loader = get_train_loader(args)

    mean_shift = ConstrainedMeanShiftSelf(
        args.arch,
        m=args.momentum,
        mem_bank_size=args.mem_bank_size,
        topk=args.topk,
        dataset_size=len(train_loader.dataset),
        topkp=args.topkp
    )
    mean_shift.data_parallel()
    mean_shift = mean_shift.cuda()
    print(mean_shift)

    params = [p for p in mean_shift.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.learning_rate,
                                momentum=args.sgd_momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    args.start_epoch = 1

    if args.weights:
        print('==> load weights from checkpoint: {}'.format(args.weights))
        ckpt = torch.load(args.weights)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        if 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt['state_dict']
        msg = mean_shift.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
        print(msg)

    if args.resume:
        print('==> resume from checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume, map_location='cpu')
        print('==> resume from epoch: {}'.format(ckpt['epoch']))

        mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
        mean_shift.queue_ptr = torch.zeros(1, dtype=torch.long)

        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
        torch.cuda.empty_cache()

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        train(epoch, train_loader, mean_shift, optimizer, args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # saving the model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': mean_shift.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            save_file = os.path.join(args.ckpt_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()


def train(epoch, train_loader, mean_shift, optimizer, opt):
    """
    one epoch training for CompReSS
    """
    mean_shift.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    purity_meter = AverageMeter()

    end = time.time()
    for idx, (indices, (im_q, im_t), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        im_q = im_q.cuda(non_blocking=True)
        im_t = im_t.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # ===================forward=====================
        loss, purity = mean_shift(im_q=im_q, im_t=im_t, labels=labels, indices=indices)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), im_q.size(0))
        purity_meter.update(purity.item(), im_q.size(0))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'purity {purity.val:.3f} ({purity.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   purity=purity_meter,
                   loss=loss_meter))
            sys.stdout.flush()
            sys.stdout.flush()

    return loss_meter.avg


if __name__ == '__main__':
    main()

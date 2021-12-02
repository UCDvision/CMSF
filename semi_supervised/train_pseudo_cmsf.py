# Train constrained mean shift in semi-supervised setting. Use GT labels for supervised set and pseudo-labels for
# unsupervised set. Pseudo-labels are obtained by training a linear layer / MLP with xent loss.

import builtins
import os
import sys
import time
import argparse
from os.path import join
import json
import pdb

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from data_loader import get_train_loader
from pseudo_cmsf import PseudoCMSF
from pseudo_label_train import train_pseudo_lbl
from util import adjust_learning_rate, AverageMeter
from tools import get_logger


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
    parser.add_argument('--sup-split-file', type=str, default=None,
                        help='path to supervised images list (text file)')
    parser.add_argument('--debug', action='store_true', help='whether in debug mode or not')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--pseudo_batch_size', type=int, default=256,
                        help='batch_size for pseudo-lbl training')
    parser.add_argument('--num_workers', type=int, default=24,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='90,120',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--cos', action='store_true',
                        help='whether to cosine learning rate or not')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--sgd_momentum', type=float, default=0.9,
                        help='SGD momentum')

    # model definition
    parser.add_argument('--arch', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18', 'resnet50', 'mobilenet'])

    # Mean Shift
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--mem_bank_size', type=int, default=128000)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--weak_strong', action='store_true',
                        help='whether to strong/strong or weak/strong augmentation')

    parser.add_argument('--weights', type=str, help='weights to initialize the model from')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--restart', action='store_true',
                        help='do not load optimizer while resuming from checkpoint')

    # Pseudo-labelling
    parser.add_argument('--topk_lbl', type=int, default=1,
                        help='top-k to be used for pseudo-labeling. All labels in top-k are used in constraint')
    parser.add_argument('--pseudo_lbl_epochs', type=int, default=40,
                        help='number of training epochs for pseudo-labelling')
    parser.add_argument('--pseudo_lbl_lr_schedule', type=str, default='15,30,40',
                        help='lr drop schedule')
    parser.add_argument('--pseudo_lbl_lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--pseudo_lbl_momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--pseudo_lbl_weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--mlp_xent', type=str, default=['linear'],
                        choices=['linear', 'mlp', 'mlp_3l'],
                        help='architecture of mlp head for xent pseudo-lbl')
    parser.add_argument('--use_conf', action='store_true',
                        help='use confidence thresholding for pseudo-labeling')
    parser.add_argument('--conf_th', type=float, default=0.,
                        help='confidence threshold value for confidence based pseudo-labeling')
    parser.add_argument('--sup_mem_bank_size', default=12800, type=int,
                        help='length of memory bank to store features and labels of supervised set')
    parser.add_argument('--cache_sup', action='store_true',
                        help='cache supervised features and labels during cmsf training, use for pseudo-label training')
    parser.add_argument('--cache_conf_unsup', action='store_true',
                        help='cache high confidence unsupervised features and labels during cmsf training')
    parser.add_argument('--use_query', action='store_true',
                        help='use query features also for psuedo-labeling')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='margin for triplet loss in CMSF')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--checkpoint_path', default='output/mean_shift_default', type=str,
                        help='where to save checkpoints. ')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def main():
    args = parse_option()
    # os.makedirs(args.checkpoint_path, exist_ok=True)

    save_dir = join(args.base_dir, 'semi_sup_cmsf/exp') 
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
    else:
        logger = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print(args)

    train_loader, train_pseudo_loader = get_train_loader(args)

    if args.dataset == 'imagenet100':
        ncls = 100
    elif args.dataset == 'imagenet':
        ncls = 1000
    else:
        sys.exit('dataset not implemented: %d' % args.dataset)

    pseudo_cmsf = PseudoCMSF(
        args.arch,
        args.mlp_xent,
        m=args.momentum,
        mem_bank_size=args.mem_bank_size,
        topk=args.topk,
        ncls=ncls,
        opt=args
    )
    pseudo_cmsf.data_parallel()
    pseudo_cmsf = pseudo_cmsf.cuda()
    print(pseudo_cmsf)

    params = [p for p in pseudo_cmsf.parameters() if p.requires_grad]
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
        # msg = pseudo_cmsf.load_state_dict(sd, strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # args.start_epoch = ckpt['epoch'] + 1
        sd = {'encoder_q.' + k: v for k, v in sd.items()}
        msg = pseudo_cmsf.load_state_dict(sd, strict=False)
        print(msg)

    if args.resume:
        print('==> resume from checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        sd = ckpt['state_dict']
        pseudo_cmsf.load_state_dict(sd, strict=True)
        if not args.restart:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = ckpt['epoch'] + 1
            # Used when memory bank is not loaded. Fill up memory bank before starting training
            # fill_mem_bank(pseudo_cmsf, train_loader, args)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        # sup_mem_bank_size >> len(sup_set) + len(high_conf_unsup_set) when unsup is also being cached. Set pointer
        # to 0 at the start of each epoch and use features till the queue pointer at the end of epoch for
        # pseudo-label training
        if args.cache_sup:
            pseudo_cmsf.sup_queue_oflow[0] = 0
            if args.cache_conf_unsup:
                pseudo_cmsf.sup_queue_ptr[0] = 0

        time1 = time.time()
        # One epoch of Pseudo-CMSF training
        _ = train(epoch, train_loader, pseudo_cmsf, optimizer, args, logger, train_pseudo_loader)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # One round of pseudo-label network training (one round = 10 epochs)
        train_pseudo_lbl(pseudo_cmsf, train_pseudo_loader, pseudo_cmsf.encoder_t, pseudo_cmsf.mlp_head, logger, args)
        time3 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time3 - time2))

        # saving the model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': pseudo_cmsf.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            save_file = os.path.join(args.ckpt_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()


def train(epoch, train_loader, pseudo_cmsf, optimizer, opt, logger, train_pseudo_loader):
    """
    one epoch training for CompReSS
    """
    pseudo_cmsf.train()
    pseudo_cmsf.mlp_head.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    purity_meter = AverageMeter()
    purity_msf_meter = AverageMeter()
    acc_pseudo_meter = AverageMeter()
    acc_topk_pseudo_meter = AverageMeter()

    end = time.time()
    for idx, (indices, (im_q, im_t), labels, is_unsup) in enumerate(train_loader):
        if (idx == len(train_loader) // 2) and (opt.batch_size < 1024):
            train_pseudo_lbl(pseudo_cmsf, train_pseudo_loader, pseudo_cmsf.encoder_t, pseudo_cmsf.mlp_head, logger, opt)
            pseudo_cmsf.train()
            pseudo_cmsf.mlp_head.eval()

        data_time.update(time.time() - end)
        im_q = im_q.cuda(non_blocking=True)
        im_t = im_t.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # ===================forward=====================
        loss, purity, purity_msf, acc_pseudo, acc_topk_pseudo, prob_topk = pseudo_cmsf(
            im_q=im_q, im_t=im_t, gt_labels=labels, is_unsup=is_unsup)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), im_q.size(0))
        purity_meter.update(purity.item(), im_q.size(0))
        purity_msf_meter.update(purity_msf.item(), im_q.size(0))
        acc_pseudo_meter.update(acc_pseudo.item(), im_q.size(0))
        acc_topk_pseudo_meter.update(acc_topk_pseudo.item(), im_q.size(0))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'purity {purity.val:.3f} ({purity.avg:.3f})\t'
                  'purity-msf {purity_msf.val:.3f} ({purity_msf.avg:.3f})\t'
                  'acc-pseudo {acc_pseudo.val:.3f} ({acc_pseudo.avg:.3f})\t'
                  'acc-topk-pseudo {acc_topk_pseudo.val:.3f} ({acc_topk_pseudo.avg:.3f})\t'
                  .format(
                   epoch, idx + 1, len(train_loader),
                   batch_time=batch_time,
                   data_time=data_time,
                   purity=purity_meter,
                   purity_msf=purity_msf_meter,
                   acc_pseudo=acc_pseudo_meter,
                   acc_topk_pseudo=acc_topk_pseudo_meter,
                   loss=loss_meter,
                   ))
            sys.stdout.flush()
            sys.stdout.flush()

    return loss_meter.avg


def fill_mem_bank(pseudo_cmsf, train_loader, opt):

    print('Filling up supervised and unsupervised memory banks')
    for idx, (indices, (im_q, im_t), labels, is_unsup) in enumerate(train_loader):
        if idx % 10 == 0:
            print(idx)

        im_q = im_q.cuda(non_blocking=True)
        im_t = im_t.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # ===================forward=====================
        _ = pseudo_cmsf(im_q=im_q, im_t=im_t, gt_labels=labels, is_unsup=is_unsup)

        del _

    return 0


if __name__ == '__main__':
    main()

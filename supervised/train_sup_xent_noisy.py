import builtins
import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets, models

from .util import adjust_learning_rate, AverageMeter, subset_classes
from .tools import get_logger, accuracy
from .dataloader import get_train_loader


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('data', type=str, help='path to dataset')
    parser.add_argument('--corrupt_split', type=str, default='25percent', required=True,
                        choices=['25percent', '50percent'],
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

    parser.add_argument('--std_aug', action='store_true',
                        help='use the standard pytorch examples augmentation instead of MoCo-v2/BYOL one')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--checkpoint_path', default='output/sup_xent_default', type=str,
                        help='where to save checkpoints. ')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


global best_acc1


def main():
    args = parse_option()
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if not args.debug:
        os.environ['PYTHONBREAKPOINT'] = '0'
        logger = get_logger(
            logpath=os.path.join(args.checkpoint_path, 'logs'),
            filepath=os.path.abspath(__file__)
        )

        def print_pass(*arg):
            logger.info(*arg)
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print(args)

    train_loader = get_train_loader(args)

    model = models.__dict__[args.arch](num_classes=1000)
    model = nn.DataParallel(model).cuda()
    print(model)

    criterion = nn.CrossEntropyLoss().cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.learning_rate,
                                momentum=args.sgd_momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    args.start_epoch = 1
    best_acc1 = 0.0

    if args.resume:
        print('==> resume from checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        model.load_state_dict(ckpt['state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
        best_acc1 = ckpt['best_acc1']

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train(epoch, train_loader, model, criterion, optimizer, args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # saving the model
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc1': best_acc1,
            }

            save_file = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()


def train(epoch, train_loader, model, criterion, optimizer, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # ===================forward=====================
        logits = model(images)
        loss = criterion(logits, labels)

        # ===================meters=====================
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1[0], images.size(0))
        acc5_meter.update(acc5[0], images.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1.val:.2f} ({acc1.avg:.2f})\t'
                  'Acc@5 {acc5.val:.2f} ({acc5.avg:.2f})\t'
                  .format(
                   epoch, idx + 1, len(train_loader),
                   batch_time=batch_time,
                   data_time=data_time,
                   acc1=acc1_meter,
                   acc5=acc5_meter,
                   loss=loss_meter))
            sys.stdout.flush()
            sys.stdout.flush()

    return loss_meter.avg


def validate(epoch, val_loader, model, criterion, optimizer, opt):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            data_time.update(time.time() - end)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # ===================forward=====================
            logits = model(images)
            loss = criterion(logits, labels)

            # ===================meters=====================
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1[0], images.size(0))
            acc5_meter.update(acc5[0], images.size(0))

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Validate: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {acc1.val:.2f} ({acc1.avg:.2f})\t'
                      'Acc@5 {acc5.val:.2f} ({acc5.avg:.2f})\t'
                      .format(
                       epoch, idx + 1, len(val_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       acc1=acc1_meter,
                       acc5=acc5_meter,
                       loss=loss_meter))
                sys.stdout.flush()
                sys.stdout.flush()

    return acc1_meter.avg


if __name__ == '__main__':
    main()


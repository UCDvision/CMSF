import argparse
import os
import sys
from os.path import join
import random
import shutil
import time
import warnings
import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

from tools import *
from util import subset_classes, adjust_learning_rate
# from utils import dict2items, read_txt_file
# from models.alexnet import AlexNet
# from models.mobilenet import MobileNetV2


parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['imagenet', 'imagenet100', 'imagenet-lt'],
                    help='use full or subset of the dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--cos', action='store_true',
                    help='whether to cosine learning rate or not')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save', default='./output/distill_1', type=str,
                    help='experiment output directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load-epoch', default=200, type=int,
                    help='epoch number of loaded model')
parser.add_argument('--train-file', type=str,
                    help='text file with train images list')
parser.add_argument('--train-lbl-file', type=str,
                    help='text file with train image labels list')
parser.add_argument('--test-file', type=str,
                    help='text file with test images list')
parser.add_argument('--test-lbl-file', type=str,
                    help='text file with test image labels list')
parser.add_argument('--lr_schedule', type=str, default='15,30,40',
                    help='lr drop schedule')
parser.add_argument('--mlp', action='store_true',
                    help='should the linear layer be a 2-layer MLP layer')
parser.add_argument('--pseudo-lbl', action='store_true',
                    help='save output labels as a list for val/test image set')
parser.add_argument('--get-train-lbls', action='store_true',
                    help='save gt labels and images as a list for train set')
parser.add_argument('--use-proj', action='store_true',
                    help='use outputs of first layer of projection head as features')
parser.add_argument('--use-cls', action='store_true',
                    help='directly use outputs of classifier for evaluation')
parser.add_argument('--load-ft-cls', action='store_true',
                    help='directly use outputs of fine-tuned classifier for evaluation')
parser.add_argument('--use-target', action='store_true',
                    help='use target encoder for evaluation')
parser.add_argument('--fine-tune', action='store_true',
                    help='train the entire network (encoder+classifier)')
parser.add_argument('--load-cache', action='store_true',
                    help='load cached values for initializing batchnorm params')
parser.add_argument('--conf-th', default=0.0, type=float,
                    help='use confidence based thresholding for pseudo-labeling')

best_acc1 = 0


def main():
    global logger

    args = parser.parse_args()
    args = parser.parse_args()
    if not os.path.exists(args.weights):
        sys.exit("Checkpoint does not exist!")
    msg = ''
    if args.fine_tune:
        msg = 'Fine-tuning the encoder!!! '
        args.save = join(args.save, 'fine_tune_bn_mlp')

    if args.use_proj:
        msg += 'Using Projection Head!!!'
        args.save = join(args.save, 'proj_head')
    elif args.use_cls:
        msg += 'Using Classifier Head!!!'
        args.save = join(args.save, 'classifier')

    args.msg = msg

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs_ep_%03d' % args.load_epoch),
                        filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def load_weights(model, wts_path, opt=None):
    wts = torch.load(wts_path)
    # pdb.set_trace()
    if 'state_dict' in wts:
        ckpt = wts['state_dict']
    elif 'model' in wts:
        ckpt = wts['model']
    else:
        ckpt = wts

    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    if opt.use_proj:
        # projection head fcs are named fc.0.weight, fc.1.weight etc, whereas using a single fc here, so it is named
        # fc.weight. change the name for compatability.
        ckpt = {k.replace('fc.mlp.0', 'fc'): v for k, v in ckpt.items()}
    if opt.use_cls:
        # load encoder_t, projection head and mlp_head (classifier)
        ckpt = {k: v for k, v in ckpt.items() if 'encoder_q' not in k}
        ckpt = {k.replace('mlp_head.', '2.'): v for k, v in ckpt.items()}
        ckpt = {k.replace('encoder_t.', '0.'): v for k, v in ckpt.items()}
    elif opt.use_target:
        ckpt = {k: v for k, v in ckpt.items() if 'encoder_q' not in k}
        ckpt = {k.replace('encoder_t.', ''): v for k, v in ckpt.items()}
    else:
        ckpt = {k: v for k, v in ckpt.items() if 'encoder_t' not in k}
        ckpt = {k.replace('encoder_q.', ''): v for k, v in ckpt.items()}

    state_dict = {}
    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print('not copied => ' + m_key)

    model.load_state_dict(state_dict)


def load_cls_weights(model, wts_path, opt=None):
    wts = torch.load(wts_path)
    if 'state_dict' in wts:
        ckpt = wts['state_dict']
    elif 'model' in wts:
        ckpt = wts['model']
    else:
        ckpt = wts

    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items() if 'classifier_q' in k}
    ckpt = {k.replace('classifier_q.mlp.', ''): v for k, v in ckpt.items()}

    state_dict = {}
    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print('not copied => ' + m_key)

    model.load_state_dict(state_dict)


def get_mlp(inp_dim, hidden_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp


class UnitNorm(nn.Module):

    def forward(self, x):
        x = nn.functional.normalize(x, dim=1)
        return x


def get_model(arch, wts_path, opt):
    if arch == 'alexnet':
        model = AlexNet()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif arch == 'pt_alexnet':
        model = models.alexnet()
        classif = list(model.classifier.children())[:5]
        model.classifier = nn.Sequential(*classif)
        load_weights(model, wts_path)
    elif arch == 'mobilenet':
        model = MobileNetV2()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif 'sup_resnet' in arch:
        model = models.__dict__[arch.replace('sup_', '')](pretrained=True)
        model.fc = nn.Sequential()
    elif 'resnet' in arch:
        model = models.__dict__[arch]()
        fc_dim = 2048
        if opt.use_proj:
            model.fc = nn.Linear(fc_dim, fc_dim * 2)
        elif opt.use_cls:
            model.fc = get_mlp(fc_dim, fc_dim * 2, fc_dim // 4)
            mlp_head = get_mlp(fc_dim // 4, fc_dim // 2, 1000)
            unit_norm = UnitNorm()
            model = nn.Sequential(model, unit_norm, mlp_head)
        # elif opt.load_ft_cls:
        #     model.fc = nn.Sequential()
            # mlp_head = get_mlp(fc_dim // 4, fc_dim // 2, 1000)
            # unit_norm = UnitNorm()
            # model = nn.Sequential(model, unit_norm, mlp_head)
        else:
            model.fc = nn.Sequential()
        load_weights(model, wts_path, opt)
    else:
        raise ValueError('arch not found: ' + arch)

    for p in model.parameters():
        p.requires_grad = False

    return model


class ImageFolderEx(datasets.ImageFolder):
    def __init__(self, root, img_file=None, lbl_file=None, *args, **kwargs):
        super(ImageFolderEx, self).__init__(root, *args, **kwargs)

        if img_file is not None and lbl_file is not None:
            with open(img_file, 'r') as f:
                imgs = [line.strip() for line in f.readlines()]
            imgs = [join(root, item.split('_')[0], item) for item in imgs]

            with open(lbl_file, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            labels = [int(float(item)) for item in labels]

            samples = []
            for (img, lbl) in zip(imgs, labels):
                samples.append((img, lbl))
            self.samples = samples

        if img_file is not None and lbl_file is None:
            with open(img_file, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            sup_set = set(lines)
            samples = []
            targets = []
            for image_path, image_class in self.samples:
                image_name = image_path.split('/')[-1]
                if image_name in sup_set:
                    samples.append((image_path, image_class))
                    targets.append(image_class)
            self.samples = samples
            self.targets = targets

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        name = self.samples[index][0]
        return name, sample, target


def main_worker(args):
    global best_acc1

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    if args.pseudo_lbl:
        valdir = os.path.join(args.data, 'train')
    else:
        valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolderEx(traindir, args.train_file, args.train_lbl_file, train_transform)
    if args.pseudo_lbl:
        val_dataset = ImageFolderEx(valdir, args.test_file, args.test_lbl_file, train_transform)
    else:
        val_dataset = ImageFolderEx(valdir, args.test_file, args.test_lbl_file, val_transform)
    train_val_dataset = ImageFolderEx(traindir, args.train_file, args.train_lbl_file, val_transform)

    # train_dataset = datasets.ImageFolder(traindir, train_transform)
    # val_dataset = datasets.ImageFolder(valdir, val_transform)
    # train_val_dataset = datasets.ImageFolder(traindir, val_transform),

    if args.dataset == 'imagenet100':
        # If label file is provided, it is assumed that the labels are for imagenet100 and the corresponding images
        # file are only from imagenet100
        if args.train_lbl_file is None:
            subset_classes(train_dataset, num_classes=100)
            subset_classes(train_val_dataset, num_classes=100)
        if args.test_lbl_file is None:
            subset_classes(val_dataset, num_classes=100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    backbone = get_model(args.arch, args.weights, args)
    backbone = nn.DataParallel(backbone).cuda()
    if not args.fine_tune:
        backbone.eval()

    cached_feats = '%s/var_mean.pth.tar' % args.save
    if not (args.use_cls or args.load_ft_cls):
        if not os.path.exists(cached_feats) or (~args.load_cache):
            train_feats, _ = get_feats(train_val_loader, backbone, args)
            train_var, train_mean = torch.var_mean(train_feats, dim=0)
            torch.save((train_var, train_mean), cached_feats)
        else:
            train_var, train_mean = torch.load(cached_feats)

    if args.mlp:
        c = get_channels(args.arch, args)
        linear = nn.Sequential(
            Normalize(),
            FullBatchNorm(train_var, train_mean),
            nn.Linear(c, c),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Linear(c, len(train_dataset.classes)),
        )
    elif args.use_cls:
        linear = nn.Sequential()
        # linear = nn.Sequential(
        #     Normalize(),
        #     FullBatchNorm(train_var, train_mean),
        #     nn.Linear(len(train_dataset.classes), len(train_dataset.classes)),
        # )
    elif args.load_ft_cls:
        c = get_channels(args.arch, args)
        linear = nn.Sequential(
            nn.BatchNorm1d(c),
            nn.Linear(c, 2 * c),
            nn.BatchNorm1d(2 * c),
            nn.ReLU(inplace=True),
            nn.Linear(2 * c, len(train_dataset.classes)),
        )
        load_cls_weights(linear, args.weights)
        linear = nn.Sequential(Normalize(), linear)
    elif args.fine_tune:
        if not args.use_proj:
            c = get_channels(args.arch, args)
            linear = nn.Sequential(
                Normalize(),
                FullBatchNorm(train_var, train_mean),
                nn.Linear(c, c),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Linear(c, len(train_dataset.classes)),
            )
        else:
            linear = nn.Sequential(
                Normalize(),
                FullBatchNorm(train_var, train_mean),
                nn.Linear(get_channels(args.arch, args), len(train_dataset.classes)),
            )
    else:
        linear = nn.Sequential(
            Normalize(),
            FullBatchNorm(train_var, train_mean),
            nn.Linear(get_channels(args.arch, args), len(train_dataset.classes)),
        )

    print(backbone)
    print(linear)
    print(args.msg)

    linear = linear.cuda()

    if args.fine_tune:
        optimizer = torch.optim.SGD([{'params': backbone.parameters(), 'lr': args.lr},
                                    {'params': linear.parameters(), 'lr': args.lr}],
                                    # args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif not args.use_cls:
        optimizer = torch.optim.SGD(linear.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if not args.use_cls:
        sched = [int(x) for x in args.lr_schedule.split(',')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sched
        )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            linear.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.get_train_lbls:
        # obtain the list of train images and corresponding labels (needed especially when original train list is
        # from imagenet while eval is on imagenet100)
        all_names = []
        all_targets = []
        for i, (names, _, targets) in enumerate(train_loader):
            all_names.append(names)
            all_targets.append(targets)
        all_names = np.concatenate(all_names)
        all_targets = np.concatenate(all_targets)
        out_file = join(args.save, '10_percent_gt_images.txt')
        np.savetxt(out_file, all_names, '%s')
        out_file = join(args.save, '10_percent_gt_lbl.txt')
        np.savetxt(out_file, all_targets, '%s')
        print('GT-lbl file saved: ', out_file)

    if args.evaluate:
        acc, pred, prob, feat, gt_lbl = validate(val_loader, backbone, linear, args, args.pseudo_lbl)
        print(acc)
        if args.pseudo_lbl:
            names, lbl = dict2items(pred)
            probs = [prob[name] for name in names]
            feats = [feat[name] for name in names]
            gt_lbls = [gt_lbl[name] for name in names]
            # pred_lbl = []
            # names_90p = read_txt_file(args.test_file)
            # if args.dataset == 'imagenet100':
                # 90percent.txt contains 90% of full imagenet, filter only those in imagenet100
                # val_cls = val_dataset.classes
                # names_90p = [item for item in names_90p if item.split('_')[0] in val_cls]

            # for name in names_90p:
            #     pred_lbl.append(pred[name])
            out_file = join(args.save, '90_percent_pseudo_conf_pt%02d_images_xent.txt' % (args.conf_th * 100))
            np.savetxt(out_file, names, '%s')
            out_file = join(args.save, '90_percent_pseudo_conf_pt%02d_lbl_xent.txt' % (args.conf_th * 100))
            np.savetxt(out_file, lbl, '%s')
            out_file = join(args.save, '90_percent_pseudo_conf_pt%02d_gt_lbl_xent.txt' % (args.conf_th * 100))
            np.savetxt(out_file, gt_lbls, '%s')
            out_file = join(args.save, '90_percent_pseudo_conf_pt%02d_prob_xent.txt' % (args.conf_th * 100))
            np.savetxt(out_file, probs, '%s')
            out_file = join(args.save, '90_percent_pseudo_conf_pt%02d_feat_xent.npy' % (args.conf_th * 100))
            np.save(out_file, feats)
            print('Pseudo-lbl file saved: ', out_file)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.cos:
            adjust_learning_rate(epoch, args, optimizer)
        # train for one epoch
        train(train_loader, backbone, linear, optimizer, epoch, args)

        # evaluate on validation set
        if epoch % 1 == 0 or epoch == (args.epoch - 1):
            acc1, _, _, _, _ = validate(val_loader, backbone, linear, args)

        # modify lr
        if not args.cos:
            lr_scheduler.step()
        # logger.info('LR: {:f}'.format(lr_scheduler.get_last_lr()[-1]))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': linear.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, is_best, args.save)


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch, opt=None):
    if arch == 'alexnet':
        c = 4096
    elif arch == 'pt_alexnet':
        c = 4096
    elif 'resnet50' in arch:
        c = 2048
    elif arch == 'resnet18':
        c = 512
    elif arch == 'mobilenet':
        c = 1280
    else:
        raise ValueError('arch not found: ' + arch)
    if opt.use_proj:
        c *= 2
    return c


def train(train_loader, backbone, linear, optimizer, epoch, args):
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
    if args.fine_tune:
        backbone.train()
    else:
        backbone.eval()
    linear.train()

    end = time.time()
    for i, (_, images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))


def validate(val_loader, backbone, linear, args, pseudo_lbl=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    all_pred = {}
    all_prob = {}
    all_feat = {}
    all_gtlbl = {}
    with torch.no_grad():
        end = time.time()
        for i, (names, images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            feats = backbone(images)
            output = linear(feats)
            prob = F.softmax(output, -1)
            prob = torch.max(prob, dim=-1)[0]
            pred = torch.argmax(output, dim=-1)
            loss = F.cross_entropy(output, target)

            # if pseudo-lbl, save predictions
            if pseudo_lbl:
                for j, name in enumerate(names):
                    if prob[j] > args.conf_th:
                        name = name.split('/')[-1]
                        all_pred[name] = int(pred[j].detach().cpu().numpy())
                        all_prob[name] = prob[j].detach().cpu().numpy()
                        all_feat[name] = feats[j].detach().cpu().numpy()
                        all_gtlbl[name] = target[j].detach().cpu().numpy()

            conf_idx = torch.ge(prob, args.conf_th)
            conf_idx = torch.where(conf_idx)
            if len(conf_idx[0]) > 0:
                output = output[conf_idx]
                target = target[conf_idx]
            else:
                print('no confident samples in iter: %d', i)
                continue
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, all_pred, all_prob, all_feat, all_gtlbl


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def get_feats(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (_, images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = normalize(model(images)).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

    return feats, labels


if __name__ == '__main__':
    main()

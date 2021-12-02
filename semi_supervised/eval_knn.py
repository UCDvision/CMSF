import builtins
import pdb
from collections import Counter, OrderedDict
from random import shuffle
import argparse
import os
from os.path import join
import sys
import random
import shutil
import time
import warnings

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
import faiss

from tools import *
from models.resnet import resnet18, resnet50


parser = argparse.ArgumentParser(description='NN evaluation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['imagenet', 'imagenet100', 'imagenet-lt'],
                    help='use full or subset of the dataset')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', type=str, default='alexnet',
                        choices=['alexnet' , 'resnet18' , 'resnet50', 'mobilenet' ,
                                 'l_resnet18', 'l_resnet50', 
                                 'two_resnet50', 'one_resnet50', 
                                 'moco_alexnet' , 'moco_resnet18' , 'moco_resnet50', 'moco_mobilenet', 'resnet50w5', 'teacher_resnet18',  'teacher_resnet50',
                                 'sup_alexnet' , 'sup_resnet18' , 'sup_resnet50', 'sup_mobilenet', 'pt_alexnet', 'swav_resnet50', 'byol_resnet50'])
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--epoch', default=200, type=int,
                    help='epoch of the model being evaluated')
parser.add_argument('--save', default='./output/cluster_alignment_1', type=str,
                    help='experiment output directory')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--train-file', type=str,
                    help='text file with train images list')
parser.add_argument('--train-lbl-file', type=str,
                    help='text file with train image labels list')
parser.add_argument('--test-file', type=str,
                    help='text file with test images list')
parser.add_argument('--test-lbl-file', type=str,
                    help='text file with test image labels list')
parser.add_argument('--save-acc', action='store_true',
                    help='save accuracy value to a file')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('--pseudo-label', action='store_true',
                    help='save output labels as a list for val/test image set')
parser.add_argument('-k', default=1, type=int, help='k in kNN')
parser.add_argument('--debug', action='store_true', help='whether in debug mode or not')


def main():
    global logger

    args = parser.parse_args()
    if not os.path.exists(args.weights):
        sys.exit("Checkpoint does not exist!")
    makedirs(args.save)

    if not args.debug:
        logger = get_logger(
            # logpath=os.path.join(args.save, 'logs'),
            logpath=os.path.join(args.save, 'knn.logs'),
            filepath=os.path.abspath(__file__)
        )
        def print_pass(*args):
            logger.info(*args)
        builtins.print = print_pass

    print(args)

    main_worker(args)


def get_model(args):

    if 'resnet' in args.arch:
        model = models.__dict__[args.arch]()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'predict' not in k}
        sd = {k: v for k, v in sd.items() if 'queue' not in k}
        sd = {k: v for k, v in sd.items() if 'labels' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_t' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {k.replace('encoder.', ''): v for k, v in sd.items()}
        sd = {('module.'+k): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        print(model)
        print(msg)

    for param in model.parameters():
        param.requires_grad = False

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
            for image_path, image_class in self.samples:
                image_name = image_path.split('/')[-1]
                if image_name in sup_set:
                    samples.append((image_path, image_class))
            self.samples = samples

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index, sample, target


def get_loaders(dataset_dir, bs, workers, dataset='imagenet', opt=None):
    traindir = os.path.join(dataset_dir, 'train')
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolderEx(traindir, opt.train_file, opt.train_lbl_file, augmentation)
    val_dataset = ImageFolderEx(valdir, opt.test_file, opt.test_lbl_file, augmentation)

    if dataset == 'imagenet100':
        subset_classes(train_dataset, num_classes=100)
        subset_classes(val_dataset, num_classes=100)


    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    return train_loader, val_loader


def main_worker(args):

    start = time.time()
    # Get train/val loader 
    # ---------------------------------------------------------------
    train_loader, val_loader = get_loaders(args.data, args.batch_size, args.workers, args.dataset, args)

    # Create and load the model
    # If you want to evaluate your model, modify this part and load your model
    # ------------------------------------------------------------------------
    # MODIFY 'get_model' TO EVALUATE YOUR MODEL
    model = get_model(args)

    # ------------------------------------------------------------------------
    # Forward training samples throw the model and cache feats
    # ------------------------------------------------------------------------
    cudnn.benchmark = True

    train_feats, train_labels, train_inds = get_feats(train_loader, model, args.print_freq)
    val_feats, val_labels, val_inds = get_feats(val_loader, model, args.print_freq)

    train_feats = l2_normalize(train_feats)
    val_feats = l2_normalize(val_feats)

    for k in [1, 5, 10, 20]:
        acc, pred, purity = faiss_knn(train_feats, train_labels, val_feats, val_labels, k)
        print(' * {}-NN Acc {:.2f}'.format(k, acc))
        print(' * {}-NN Purity {:.2f}'.format(k, purity))
        if args.save_acc:
            np.savetxt(join(args.save, 'nn_%d_acc_epoch_%03d.txt' % (k, args.epoch)), [acc])
            np.savetxt(join(args.save, 'nn_%d_purity_epoch_%03d.txt' % (k, args.epoch)), [purity])
        if args.pseudo_label:
            out_file = './features/' \
                       '/90percent_pseudo_lbl_%d_nn.txt' % k
            # torch.save(pred, out_file, _use_new_zipfile_serialization=False)
            np.savetxt(out_file, pred, '%s')


def l2_normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def faiss_knn(feats_train, targets_train, feats_val, targets_val, k):
    feats_train = feats_train.numpy()
    targets_train = targets_train.numpy()
    feats_val = feats_val.numpy()
    targets_val = targets_val.numpy()

    d = feats_train.shape[-1]

    index = faiss.IndexFlatL2(d)  # build the index
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(feats_train)

    D, I = gpu_index.search(feats_val, k)

    pred = np.zeros(I.shape[0], dtype=np.int32)
    # percentage of correct class in top-k neighbours
    purity = np.zeros(I.shape[0], dtype=np.int32)
    conf_mat = np.zeros((1000, 1000), dtype=np.int32)
    for i in range(I.shape[0]):
        votes = list(Counter(targets_train[I[i]]).items())
        shuffle(votes)
        pred[i], purity[i] = max(votes, key=lambda x: x[1])
        topk_cls = [item[0] for item in votes]
        try:
            cls_idx = topk_cls.index(targets_val[i])
            purity[i] = votes[cls_idx][1]
        except ValueError:
            purity[i] = 0
        conf_mat[targets_val[i], pred[i]] += 1

    acc = 100.0 * (pred == targets_val).mean()
    purity = (purity / (k * 1.)).mean() * 100.
    assert acc == (100.0 * (np.trace(conf_mat) / np.sum(conf_mat)))

    return acc, pred, purity


def get_feats(loader, model, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, indices, ptr = None, None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (index, images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = model(images).cpu()
            cur_indices = index.cpu()

            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()
                indices = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            indices.index_copy_(0, inds, cur_indices)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(progress.display(i))

    return feats, labels, indices


def subset_classes(dataset, num_classes=10):
    np.random.seed(1234)
    all_classes = sorted(dataset.class_to_idx.items(), key=lambda x: x[1])
    subset_classes = [all_classes[i] for i in np.random.permutation(len(all_classes))[:num_classes]]
    subset_classes = sorted(subset_classes, key=lambda x: x[1])
    dataset.classes_to_idx = {c: i for i, (c, _) in enumerate(subset_classes)}
    dataset.classes = [c for c, _ in subset_classes]
    orig_to_new_inds = {orig_ind: new_ind for new_ind, (_, orig_ind) in enumerate(subset_classes)}
    dataset.samples = [(p, orig_to_new_inds[i]) for p, i in dataset.samples if i in orig_to_new_inds]



if __name__ == '__main__':
    main()


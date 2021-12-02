import time

import torch
import torch.nn.functional as F

from util import AverageMeterV2, ProgressMeter, accuracy


def train_pseudo_lbl(pseudo_cmsf, train_val_loader, backbone, model, logger, opt):
    """Train pseudo-labeling model (mlp) till convergence.

    """
    # switch to train mode
    backbone.eval()
    model.train()

    if not opt.cache_sup:
        # Get features for all samples if they were not cached during CMSF training
        train_feats, train_labels = get_feats(backbone, train_val_loader, logger, opt)
    else:
        # Load features cached during CMSF training epoch
        train_feats = pseudo_cmsf.sup_queue.clone().detach()
        train_labels = pseudo_cmsf.sup_labels.clone().detach()
        # sup_mem_bank_size >> len(sup_set) + len(high_conf_unsup_set) when unsup is also being cached. Select only
        # till the queue pointer, remaining features are not meaningful.
        if (pseudo_cmsf.sup_queue_oflow == 0) and opt.cache_conf_unsup:
            train_feats = train_feats[:pseudo_cmsf.sup_queue_ptr]
            train_labels = train_labels[:pseudo_cmsf.sup_queue_ptr]
        if logger is not None:
            logger.info('Num samples for pseudo-label training: {:d}'.format(len(train_labels)))

    # Create a new optimizer to train the MLP classifier for each round of pseudo-label training since the optim
    # parameters would need to be reset at the end of each round.
    optimizer = torch.optim.SGD(model.parameters(),
                                opt.pseudo_lbl_lr,
                                momentum=opt.pseudo_lbl_momentum,
                                weight_decay=opt.pseudo_lbl_weight_decay)

    sched = [int(x) for x in opt.pseudo_lbl_lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )
    for epoch in range(opt.pseudo_lbl_epochs):

        batch_time = AverageMeterV2('Time', ':6.3f')
        data_time = AverageMeterV2('Data', ':6.3f')
        losses = AverageMeterV2('Loss', ':.4e')
        top1 = AverageMeterV2('Acc@1', ':6.2f')
        top5 = AverageMeterV2('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_feats)//opt.pseudo_batch_size,
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        # for i, (images, target) in enumerate(train_loader):
        n_batches = len(train_feats) // opt.pseudo_batch_size
        indices = torch.arange(len(train_feats))
        perm = torch.randperm(len(train_feats))
        indices = indices[perm]
        # for i, idx in enumerate(indices):
        for idx in range(n_batches):
            # measure data loading time
            data_time.update(time.time() - end)

            ids = indices[(idx * opt.pseudo_batch_size): (idx + 1) * opt.pseudo_batch_size]
            if opt.cache_sup:
                output = train_feats[ids]
                target = train_labels[ids]
                # Some labels in queue might be set to -1 in first epoch
                target[target == -1] = 0
            else:
                output = train_feats[ids].cuda()
                target = train_labels[ids].cuda()
            output = model(output)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), ids.size(0))
            top1.update(acc1[0], ids.size(0))
            top5.update(acc5[0], ids.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0 and logger is not None:
                logger.info(progress.display(idx))
            elif idx % opt.print_freq == 0:
                progress.display(idx)

        lr_scheduler.step()


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def get_feats(model, loader, logger, opt):
    """Obtain features and labels for all samples in data loader using current model.

    """
    batch_time = AverageMeterV2('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (_, images, target, _) in enumerate(loader):
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

            if i % opt.print_freq == 0 and logger is not None:
                logger.info(progress.display(i))
            elif i % opt.print_freq == 0:
                progress.display(i)

    return feats, labels

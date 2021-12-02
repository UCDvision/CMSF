import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet as resnet
from mlp_arch import get_mlp, get_mlp_3l


class PseudoCMSF(nn.Module):
    def __init__(self, arch, mlp_xent, m=0.99, mem_bank_size=128000, topk=5, ncls=1000, opt=None):
        super(PseudoCMSF, self).__init__()

        # save parameters
        self.m = m
        self.mem_bank_size = mem_bank_size
        self.topk = topk

        # additional parameters for pseudo-labeled constrained mean shift
        self.topk_lbl = opt.topk_lbl
        self.use_conf = opt.use_conf
        self.conf_th = opt.conf_th
        self.sup_mem_bank_size = opt.sup_mem_bank_size
        self.cache_sup = opt.cache_sup
        self.cache_conf_unsup = opt.cache_conf_unsup

        if self.topk_lbl > 1:
            sys.exit('Not Implemented!!!')
            
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

        # mlp head to perform pseudo-labeling
        if mlp_xent == 'linear':
            self.mlp_head = nn.Sequential(
                # FullBatchNorm(train_var, train_mean),
                nn.Linear(proj_dim, ncls),
            )
        elif mlp_xent == 'mlp':
            self.mlp_head = get_mlp(proj_dim, 2 * proj_dim, ncls)
        elif mlp_xent == 'mlp_3l':
            self.mlp_head = get_mlp_3l(proj_dim, 2 * proj_dim, 2 * proj_dim, ncls)

        # copy query encoder weights to target encoder
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False

        print("using mem-bank size {}".format(self.mem_bank_size))
        # setup queue (For Storing Random Targets)
        self.register_buffer('queue', torch.randn(self.mem_bank_size, proj_dim))
        # normalize the queue embeddings
        self.queue = nn.functional.normalize(self.queue, dim=1)
        # initialize the labels queue (For constrained mean shift - use pseudo labels)
        self.register_buffer('labels', -1*torch.ones(self.mem_bank_size).long())
        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # initialize the ground truth labels queue (For Purity measurement)
        self.register_buffer('gt_labels', -1 * torch.ones(self.mem_bank_size).long())

        if self.cache_sup:
            # Queue for features and labels of supervised set (and possibly highly confident unsupervised set) - caching
            # these features makes xent pseudo-label training faster
            # setup queue (For Storing Supervised Targets)
            self.register_buffer('sup_queue', torch.randn(self.sup_mem_bank_size, proj_dim))
            # normalize the queue embeddings
            self.sup_queue = nn.functional.normalize(self.sup_queue, dim=1)
            # initialize the supervised labels queue
            self.register_buffer('sup_labels', -1 * torch.ones(self.sup_mem_bank_size).long())
            # setup the supervised queue pointer
            self.register_buffer('sup_queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer('sup_queue_oflow', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def data_parallel(self):
        self.encoder_q = torch.nn.DataParallel(self.encoder_q)
        self.encoder_t = torch.nn.DataParallel(self.encoder_t)
        self.predict_q = torch.nn.DataParallel(self.predict_q)
        self.mlp_head = torch.nn.DataParallel(self.mlp_head)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, targets, labels, gt_labels):
        batch_size = targets.shape[0]

        ptr = int(self.queue_ptr)
        assert self.mem_bank_size % batch_size == 0

        # replace the targets at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = targets
        self.labels[ptr:ptr + batch_size] = labels
        self.gt_labels[ptr:ptr + batch_size] = gt_labels
        ptr = (ptr + batch_size) % self.mem_bank_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _sup_dequeue_and_enqueue(self, targets, labels):
        batch_size = targets.shape[0]

        ptr = int(self.sup_queue_ptr)

        # sup_mem_bank_size need not be a multiple of batch size.
        # if all sup data points won't fit in the queue
        # we slide back the pointer to accommodate them
        if (ptr + batch_size) > self.sup_mem_bank_size:
            ptr = self.sup_mem_bank_size - batch_size
            self.sup_queue_oflow[0] = 1

        # replace the targets at ptr (dequeue and enqueue)
        self.sup_queue[ptr:ptr + batch_size] = targets
        self.sup_labels[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.sup_mem_bank_size  # move pointer

        self.sup_queue_ptr[0] = ptr

    @torch.no_grad()
    def _pseudo_label(self, feat_t, feat_q):
        # pseudo-label prediction
        output_t = self.mlp_head(feat_t)
        prob = F.softmax(output_t, -1)
        pred = torch.argmax(prob, dim=1)
        prob_topk, pred_topk = prob.topk(5)
        return pred, pred_topk, prob_topk

    def forward(self, im_q, im_t, gt_labels, is_unsup):
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

            # Calculate pseudo labels
            pseudo_labels, pseudo_labels_topk, prob_topk = self._pseudo_label(current_target, query)

            # Replace pseudo-labels with gt labels for supervised set
            sup_idx = torch.where(is_unsup == 0)
            unsup_idx = torch.where(is_unsup == 1)
            pseudo_labels_topk[sup_idx[0], 0] = gt_labels[sup_idx]
            assert torch.all(pseudo_labels_topk[sup_idx[0], 0] == gt_labels[sup_idx])

            # If using confidence, set the labels of samples with conf < thresh to -1
            if self.use_conf:
                not_conf = torch.lt(prob_topk[:, 0], self.conf_th)
                # Supervised samples are set to confident
                not_conf[sup_idx] = False
                not_conf_idx = torch.where(not_conf)
                # conf_idx ==> idx of supervised and highly confident unsupervised samples
                conf_idx = torch.where(~not_conf)
                pseudo_labels_topk[not_conf_idx] = -1

            self._dequeue_and_enqueue(current_target, pseudo_labels_topk[:, 0], gt_labels)
            if self.cache_sup:
                if self.cache_conf_unsup:
                    # Cache both supervised and highly confident unsupervised samples
                    self._sup_dequeue_and_enqueue(current_target[conf_idx], pseudo_labels_topk[conf_idx][:, 0])
                else:
                    # Cache only supervised samples
                    self._sup_dequeue_and_enqueue(current_target[sup_idx], gt_labels[sup_idx])
                    
            # The top-k labels below threshold confidence must not participate in constraint calculation. Since the
            # label -1 is part of target memory bank labels, set the labels for non-confidant samples to -2.
            if self.use_conf:
                pseudo_labels_topk[not_conf_idx] = -2
                # pseudo_labels_topk[not_conf_idx[0], not_conf_idx[1]] = -2

        Q = query
        K = current_target
        M = self.queue.clone().detach()

        # Regardless of self.topk_lbl, pseudo_labels_topk is always set to [bs, 5] for easier purity calculation.
        # Select top-k pseudo-labels for query batch and target memory bank
        Lx = pseudo_labels_topk[:, :self.topk_lbl]
        Lm = self.labels.clone().detach()
        Lx_gt = gt_labels
        Lm_gt = self.gt_labels.clone().detach()

        b = Q.shape[0]
        m = M.shape[0]
        k = self.topk
        k2 = self.topk_lbl

        # 1. reshape labels to have same size (tile/copy labels to the expanded dim)
        Lx1 = Lx.unsqueeze(1).expand((b, m, k2))
        Lm1 = Lm.unsqueeze(0).unsqueeze(2).expand((b, m, k2))
        # Mask is 1 if none of q-top-k labels in query is present in any of t-top-k labels in target
        # i.e, mask=1 implies the target sample is from a different class compared to query
        Msk = torch.all(Lx1 != Lm1, dim=-1)
        if self.use_conf:
            # If none of the top-k pseudo-labels for a sample is confident, fallback to msf loss
            Msk[not_conf_idx] = False

        # reshape gt labels; used in purity calculation
        Lx1_gt = Lx_gt.unsqueeze(1).expand((b, m))
        Lm1_gt = Lm_gt.unsqueeze(0).expand((b, m))

        # 2. calculate distances
        Dk = 2 - 2 * (K @ M.T)
        Dq = 2 - 2 * (Q @ M.T)

        # top-k indices from Dk without class constraint (plain mean shift)
        _, iNDk_msf = Dk.topk(k, dim=1, largest=False)

        # 3. set non category distances to 5 (max distance)
        Dk[torch.where(Msk)] = 5.0

        # 4. select indices of topk distances from Dk
        _, iNDk = Dk.topk(k, dim=1, largest=False)

        # 5. using above indices, gather the distances from Dq
        NDq = torch.gather(Dq, 1, iNDk)

        # 6. first, average over k, and then average over b
        L = NDq.mean(dim=1).mean()

        # calculate purity, based on true labels for unsupervised set
        P = torch.gather(Lx1_gt == Lm1_gt, 1, iNDk)
        purity = 100 * (P.float().sum(dim=1) / k).mean()

        # purity of top-k neighbours without constraint --> purity of simple mean shift
        P = torch.gather(Lx1_gt == Lm1_gt, 1, iNDk_msf)
        purity_msf = 100 * (P.float().sum(dim=1) / k).mean()

        # calculate accuracy of pseudo-labelling (remove supervised set from pseudo-labels)
        if not self.use_conf:
            acc = (gt_labels[unsup_idx] == pseudo_labels_topk[unsup_idx[0], 0]).float().mean()
            acc_topk = torch.eq(pseudo_labels_topk[unsup_idx],
                                gt_labels[unsup_idx].unsqueeze(1).expand((-1, 5))).float()
            acc_topk = acc_topk.sum(dim=1).mean()
        else:
            # Do not consider non-confident labels for purity calculation
            acc_topk = torch.eq(pseudo_labels_topk[unsup_idx],
                                gt_labels[unsup_idx].unsqueeze(1).expand((-1, 5))).float()
            # acc is considered 0 where lbl is not confident. however, they are ignored by summing over only the
            # confident ones.
            acc_topk[:, 0] = acc_topk[:, 0] * (~not_conf[unsup_idx] * 1.)
            acc = acc_topk[:, 0].sum() / ((~not_conf[unsup_idx] * 1.).sum() + 1e-6)
            acc_topk = acc_topk.mean()
        acc *= 100.
        acc_topk *= 100.

        return L, purity, purity_msf, acc, acc_topk, prob_topk


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds



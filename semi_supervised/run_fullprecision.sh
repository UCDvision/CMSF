#!/usr/bin/env bash

set -x
set -e

base_dir='./'
exp_name='exp_semi_sup_fullprec_1'
dataset_path='path/to/imagenet/dataset'

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_pseudo_cmsf.py \
   --base-dir $base_dir \
   --exp $exp_name\
   --learning_rate 0.05\
   --batch_size 256\
   --use_conf \
   --conf_th 0.85\
   --cache_sup\
   --pseudo_lbl_epochs 10\
   --pseudo_lbl_lr_schedule '15,30,40'\
   --margin 0.3\
   --dataset imagenet \
   --cos \
   --weak_strong \
   --epochs 200 \
   --arch resnet50 \
   --mlp_xent mlp \
   --topk 10 \
   --momentum 0.99 \
   --mem_bank_size 128000 \
   --sup_mem_bank_size 128000 \
   --save_freq 10\
   --sup-split-file 'imagenet_subsets/1p_10p/subsets/10percent.txt' \
   $dataset_path

exp="$base_dir/semi_sup_cmsf/exp/$exp_name"
ep=200

CUDA_VISIBLE_DEVICES=0,1 python eval_linear.py\
    -j 16 \
    --lr 0.001\
    --lr_schedule '15,35,50'\
    --epochs 20\
    --load-ft-cls\
    --fine-tune\
    --arch 'resnet50'\
    --load-epoch $ep\
    --dataset imagenet\
    --train-file 'imagenet_subsets/1p_10p/subsets/10percent.txt'\
    --weights $exp/checkpoints/ckpt_epoch_$ep.pth\
    --save $exp/linear\
    $dataset_path
echo $exp
echo 'epoch: '$ep

CUDA_VISIBLE_DEVICES=0,1 python eval_knn.py\
    --arch 'resnet50'\
    --epoch $ep\
    --dataset imagenet\
    --batch-size 256\
    -k 1\
    --weights $exp/checkpoints/ckpt_epoch_$ep.pth\
    --train-file 'imagenet_subsets/1p_10p/subsets/10percent.txt'\
    --save $exp/features \
    --save-acc \
    $dataset_path
echo $exp
echo 'epoch: '$ep

res_dir="$exp/results"
mkdir "$res_dir"

cp "$exp/linear/logs_ep_$ep" "$res_dir"
cp "$exp/features/nn_*_acc_epoch_$ep.txt" "$res_dir"

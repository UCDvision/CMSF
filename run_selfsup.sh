
base_dir='./'
exp_name='exp_semi_sup_fullprec_1'
#dataset_path='path/to/imagenet/dataset'
dataset_path='/datasets/imagenet'

python -m self_supervised.train_cmsf_self \
    --base-dir $base_dir \
    --exp $exp_name\
    --cos \
    --weak_strong \
    --learning_rate 0.05 \
    --epochs 200 \
    --arch resnet50 \
    --topk 5 \
    --momentum 0.99 \
    --mem_bank_size 128000 \
    --topkp 5 \
    $dataset_path

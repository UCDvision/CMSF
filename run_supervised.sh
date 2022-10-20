
base_dir='./'
exp_name='exp_semi_sup_fullprec_1'
#dataset_path='path/to/imagenet/dataset'
dataset_path='/datasets/imagenet'

python -m supervised.train_cmsf_sup \
    --base-dir $base_dir \
    --exp $exp_name\
    --cos \
    --weak_strong \
    --learning_rate 0.05 \
    --epochs 200 \
    --arch resnet50 \
    --topk 10 \
    --momentum 0.99 \
    --mem_bank_size 128000 \
    $dataset_path

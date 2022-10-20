# Constrained Mean Shift (CMSF)
Official Code for the [paper](https://arxiv.org/abs/2112.04607) "Constrained Mean Shift Using Distant Yet Related Neighbors for Representation Learning".
Paper accepted at _European Conference on Computer Vision (ECCV), 2022_

<p align="center">
  <img src="https://ucdvision.github.io/CMSF/assets/images/cmsf_teaser.gif" width="95%">
</p>


## Overview
CMSF extends a prior self-supervised representation learning method(MSF) where a sample is encouraged to be close to not just its augmented version but also to the nearest neighbors of the augmented image. In CMSF, the neighbors are constrained to be from the same semantic category as the input image. Use of constraint provides samples that are far from the target image in the feature space but close in the semantic space. The category labels are present in the supervised set-up and are predicted in the semi- and self-supervised set-ups.

## Requirements

All our experiments use the PyTorch library. We recommend installing the following package versions:
- python=3.7.6
- pytorch=1.4
- torchvision=0.5.0
- faiss-gpu=1.6.1 (required for k-NN evaluation alone)

Instructions for PyTorch installation can be found [here](https://pytorch.org/). 

GPU version of the FAISS ([https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)) package is necessary for k-NN evaluation of trained models. It can be installed using the following command:
```shell
pip install faiss-gpu
```

## Training and Evaluating Semi-supervised CMSF
Modify the arguments in the ```run_semisup_fullprecision.sh``` file and run the following command:
```shell
bash run_semisup_fullprecision.sh
```
The script includes code for training the semi-supervised version and performing the k-NN and linear evaluation on the final checkpoint. Modify the arguments to run only the training / testing codes.


## Training Self-supservised CMSF

Following command can be used to train Self-supervised CMSF: 
```
python self_supervised/train_msf_2q.py \
  --cos \
  --weak_strong \
  --learning_rate 0.05 \
  --epochs 200 \
  --arch resnet50 \
  --topk 5 \
  --momentum 0.99 \
  --mem_bank_size 128000 \
  --topkp 5 \
  --checkpoint_path <CHECKPOINT PATH> \
  <DATASET PATH>
  
```


## Training Supservised CMSF

Following command can be used to train Supervised CMSF: 

```
python supervised/train_sup_msf.py \
  --cos \
  --weak_strong \
  --learning_rate 0.05 \
  --epochs 200 \
  --arch resnet50 \
  --topk 10 \
  --momentum 0.99 \
  --mem_bank_size 128000 \
  --checkpoint_path <CHECKPOINT PATH> \
  <DATASET PATH>
  
```

## TODO

- Add multi-crop codes.
- Add results and pretrained models.

## Citation

If you make use of the code, please cite the following work:

```
@inproceedings{navaneet2022constrained,
      title={Constrained Mean Shift Using Distant Yet Related Neighbors for Representation Learning}, 
      author={KL Navaneet and Soroush Abbasi Koohpayegani and Ajinkya Tejankar and Kossar Pourahmadi and Akshayvarun Subramanya and Hamed Pirsiavash},
      year={2022},
      booktitle={European Conference on Computer Vision (ECCV)}
}

```

## License

This project is under the MIT license.

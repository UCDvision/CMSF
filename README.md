# CMSF
Official Code for the [paper](https://arxiv.org/abs/2112.04607) "Constrained Mean Shift Using Distant Yet Related Neighbors for Representation Learning".
Paper accepted at _European Conference on Computer Vision (ECCV), 2022_

<p align="center">
  <img src="https://ucdvision.github.io/CMSF/assets/images/cmsf_teaser.gif" width="95%">
</p>



```
@inproceedings{navaneet2022constrained,
      title={Constrained Mean Shift Using Distant Yet Related Neighbors for Representation Learning}, 
      author={KL Navaneet and Soroush Abbasi Koohpayegani and Ajinkya Tejankar and Kossar Pourahmadi and Akshayvarun Subramanya and Hamed Pirsiavash},
      year={2022},
      booktitle={European Conference on Computer Vision (ECCV)}
}

```

# Requirements

- Python >= 3.7.6
- PyTorch >= 1.4
- torchvision >= 0.5.0
- faiss-gpu >= 1.6.1

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). We used Python 3.7 for our experiments.


- Install PyTorch ([pytorch.org](http://pytorch.org))


You will need to install FAISS to run k-NN evaluation code.

FAISS: 
- Install FAISS ([https://github.com/facebookresearch/faiss/blob/master/INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md))

# Training and Evaluating Semi-supervised CMSF
Modify the arguments in the ```run_semisup_fullprecision.sh``` file and run the following command:
```shell
bash run_semisup_fullprecision.sh
```
The script includes code for training the semi-supervised version and performing the k-NN and linear evaluation on the final checkpoint. Modify the arguments to run only the training / testing codes.


# Training Self-supservised CMSF

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


# Training Supservised CMSF

Following command can be used to train the CMSF(Supervised Learning) 

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

# License

This project is under the MIT license.

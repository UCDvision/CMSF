# CMSF
Official Code for "Constrained Mean Shift Using Distant Yet Related Neighbors for Representation Learning"

<p align="center">
  <img src="https://ucdvision.github.io/CMSF/assets/images/cmsf_teaser.gif" width="95%">
</p>



```
@misc{tejankar2021constrained,
      title={Constrained Mean Shift Using Distant Yet Related Neighbors for Representation Learning}, 
      author={Ajinkya Tejankar and Soroush Abbasi Koohpayegani and KL Navaneet and Kossar Pourahmadi and Akshayvarun Subramanya and Hamed Pirsiavash},
      year={2021},
      eprint={2112.04607},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

# Requirements

- Python >= 3.7.6
- PyTorch >= 1.4
- torchvision >= 0.5.0
- faiss-gpu >= 1.6.1

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). We used Python 3.7 for our experiments.


- Install PyTorch ([pytorch.org](http://pytorch.org))


To run NN and CMSF-KM, you require to install FAISS. 

FAISS: 
- Install FAISS ([https://github.com/facebookresearch/faiss/blob/master/INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md))


# Training Self-Supservised CMSF-KM

```
python self_supervised/train_msf_km.py \
  --cos \
  --weak_strong \
  --learning_rate 0.05 \
  --epochs 200 \
  --arch resnet50 \
  --topk 5 \
  --momentum 0.99 \
  --mem_bank_size 128000 \
  --num_clusters 50000 \
  --checkpoint_path <CHECKPOINT PATH> \
  <DATASET PATH>
  
```

  
  

# Training Self-Supservised CMSF-2Q

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



# Training Supservised 

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

import os

import random
import torch
from torchvision import transforms, datasets
from PIL import ImageFilter

from .util import subset_classes


# Extended version of ImageFolder to return index of image too.
class ImageFolderEx(datasets.ImageFolder):
    # def __init__(self, root, sup_split_file, only_sup, *args, **kwargs):
        # super(ImageFolderEx, self).__init__(root, *args, **kwargs)
    def __init__(self, root, transforms, sup_split_file=None, only_sup=False, corrupt_split_file=None):
        super(ImageFolderEx, self).__init__(root, transforms)

        self.is_unsup = None
        # Supervised subset for semi-supervised learning
        if sup_split_file is not None:
            with open(sup_split_file, 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            sup_set = set(lines)
            samples = []
            self.is_unsup = -1 * torch.ones((len(self.samples)), dtype=torch.int)
            for i, (image_path, image_class) in enumerate(self.samples):
                image_name = image_path.split('/')[-1]
                if image_name in sup_set:
                    self.is_unsup[i] = 0
                    if only_sup:
                        samples.append((image_path, image_class))
                else:
                    self.is_unsup[i] = 1

            # Use only supervised images
            if only_sup:
                self.samples = samples

        if corrupt_split_file is not None:
            with open(corrupt_split_file, 'r') as f:
                samples = [line.strip().split(' ') for line in f.readlines()]

            self.samples = [(os.path.join(root, pth), int(cls)) for pth, cls in samples]

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        if self.is_unsup is not None:
            is_unsup = self.is_unsup[index]
            return index, sample, target, is_unsup
        else:
            return index, sample, target


class TwoCropsTransform:
    """Return two random crops of one image as the query and target."""
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        print(self.weak_transform)
        print(self.strong_transform)

    def __call__(self, x):
        q = self.strong_transform(x)
        t = self.weak_transform(x)
        return [q, t]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# Create train loader
def get_train_loader(opt):
    traindir = os.path.join(opt.data, 'train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentation_strong = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_weak = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    if 'sup_split_file' in vars(opt).keys():
        sup_split_file = opt.sup_split_file
    else:
        sup_split_file = None

    if 'corrupt_split' in vars(opt).keys():
        corrupt_split_file = os.path.join('subsets', 'corrupt_{}.txt'.format(opt.corrupt_split))
    else:
        corrupt_split_file = None

    if opt.weak_strong:
        train_dataset = ImageFolderEx(
            root=traindir,
            transforms=TwoCropsTransform(transforms.Compose(augmentation_weak), transforms.Compose(augmentation_strong)),
            sup_split_file=sup_split_file,
            only_sup=False,
            corrupt_split_file=corrupt_split_file,
        )
    else:
        train_dataset = ImageFolderEx(
            root=traindir,
            transforms=TwoCropsTransform(transforms.Compose(augmentation_strong), transforms.Compose(augmentation_strong)),
            sup_split_file=sup_split_file,
            only_sup=False,
            corrupt_split_file=corrupt_split_file,
        )

    if opt.dataset == 'imagenet100':
        subset_classes(train_dataset, num_classes=100)

    print('==> train dataset')
    print(train_dataset)

    # NOTE: remove drop_last
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    # Applicable only for semi-supervised setup.
    if 'sup_split_file' in vars(opt).keys():
        # Get dataloader for pseudo-labelling
        sup_val_dataset = ImageFolderEx(traindir, opt.sup_split_file, True, transforms.Compose(augmentation_weak))
        if opt.dataset == 'imagenet100':
            subset_classes(sup_val_dataset, num_classes=100)
        train_val_loader = torch.utils.data.DataLoader(
            sup_val_dataset,
            batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True,
        )

        return train_loader, train_val_loader
    else:
        return train_loader



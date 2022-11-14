
import os

from random import random

from functools import partial

import numpy as np

import torch
from torch import nn
from torch.utils import data

import torch.nn.functional as F
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms
from stylegan2_pytorch.version import __version__
from stylegan2_pytorch.diff_augment import DiffAugment
from stylegan2_pytorch.utils import *

from PIL import Image
from pathlib import Path


def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    imgsize = image.size() if isinstance(image, torch.Tensor) else image.size
    if max(imgsize) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image



# augmentations

def random_hflip(tensor, prob):
    if prob < random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)



class FolderDataset(data.Dataset):
    def __init__(self, folder, image_size, rgb='rgb', aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        #convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = RGB_CHANNELS[rgb]

        self.transform = transforms.Compose([
            #transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            #transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)



class VGGFaceDataset(data.Dataset):
    @classmethod
    def load(cls, data_path, **kwargs):
        all_imgs = torch.tensor(np.load(data_path))
        all_imgs = all_imgs.permute(0,1,4,2,3)                          #  X x Y x C -> C x X x Y
        all_imgs = all_imgs.index_select(2, torch.tensor([2,1,0]))      #  convert from BGR to RGB
        return cls(all_imgs.view(-1, *all_imgs.size()[-3:]), **kwargs)

    def __init__(self, dataset, image_size, rgb='rgb', aug_prob = 0.):
        super().__init__()
        self.dataset = dataset
        self.image_size = image_size

        #convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = RGB_CHANNELS[rgb]

        self.transform = transforms.Compose([
            #transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            #transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return self.dataset.size(0)

    def __getitem__(self, index):
        img = self.dataset[index]
        return self.transform(img)



def make_dataset(dataset_name, imgsize, aug_prob, data_dir='./'):
    if dataset_name == "omniglot":
        dataset = FolderDataset(os.path.join(data_dir, "omniglot"), imgsize, rgb = 'bw', aug_prob = aug_prob)
    elif dataset_name == "flowers":
        dataset = FolderDataset(os.path.join(data_dir, "flowers"), imgsize, rgb = 'rgb', aug_prob = aug_prob)
    elif dataset_name == "vggface":
        dataset = VGGFaceDataset.load(os.path.join(data_dir, "vgg_face_data.npy"), imgsize, rgb = 'rgb', aug_prob = aug_prob)
    
    return dataset

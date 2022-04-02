"""
Code adapted from: https://github.com/wvangansbeke/Unsupervised-Classification
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
# from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob


class ImageNet(datasets.ImageFolder):
    def __init__(self, root="/path/to/imagenet", split='train', transform=transforms.ToTensor()):
        super(ImageNet, self).__init__(root=os.path.join(root, '%s' % (split)), transform=transforms.ToTensor())
        a = os.path.join(root, '%s' % (split))
        self.transform = transform
        self.split = split
        self.resize = tf.Resize(256)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = [img, target]

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img

    def get_loader(self):
        return torch.utils.data.DataLoader(self, shuffle=self.split == 'train', collate_fn=collate_custom,)


class ImageNetSubset(data.Dataset):

    def __init__(self, subset_file, root="/path/to/imagenet/", split='train',
                 transform=transforms.ToTensor()):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root, '%s' % (split))
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            # subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i))
        self.imgs = imgs
        self.classes = class_names

        # Resize
        self.resize = tf.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        # im_size = img.size
        img = self.resize(img)
        # class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        # out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}
        out = [img, target]

        return out

    def get_loader(self):
        return torch.utils.data.DataLoader(self, shuffle=self.split == 'train', collate_fn=collate_custom,)


def collate_custom(batch):
    """ Custom collate function """
    import numpy as np
    import collections
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], str):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))

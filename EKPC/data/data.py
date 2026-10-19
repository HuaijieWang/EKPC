from pathlib import Path

import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    data_root = None


def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform
    t = []
    if resize_im:
        size = int(256 / 224 * input_size)
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    return t


class iCIFAR224(iData):
    use_path = False
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(
            self.data_root or "./data", train=True, download=True
        )
        test_dataset = datasets.cifar.CIFAR100(
            self.data_root or "./data", train=False, download=True
        )
        self.train_data, self.train_targets = (
            train_dataset.data,
            np.array(train_dataset.targets),
        )
        self.test_data, self.test_targets = (
            test_dataset.data,
            np.array(test_dataset.targets),
        )


class iImageNetR(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = (
            str(Path(self.data_root) / "train")
            if self.data_root
            else "data/datasets/CIL/imagenet-r/train"
        )
        test_dir = (
            str(Path(self.data_root) / "test")
            if self.data_root
            else "data/datasets/CIL/imagenet-r/test"
        )
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        if train_dset.class_to_idx != test_dset.class_to_idx:
            raise ValueError("Training and evaluation class directories must match")
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = (
            str(Path(self.data_root) / "train")
            if self.data_root
            else "data/datasets/CIL/imagenet-a/imagenet-a/train/"
        )
        test_dir = (
            str(Path(self.data_root) / "test")
            if self.data_root
            else "data/datasets/CIL/imagenet-a/imagenet-a/test/"
        )
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        if train_dset.class_to_idx != test_dset.class_to_idx:
            raise ValueError("Training and evaluation class directories must match")
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class CUB(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = (
            str(Path(self.data_root) / "train")
            if self.data_root
            else "data/datasets/CIL/cub/train/"
        )
        test_dir = (
            str(Path(self.data_root) / "test")
            if self.data_root
            else "data/datasets/CIL/cub/test/"
        )
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        if train_dset.class_to_idx != test_dset.class_to_idx:
            raise ValueError("Training and evaluation class directories must match")
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class Domainnet(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = (
            str(Path(self.data_root) / "train")
            if self.data_root
            else "data/datasets/CIL/domainnet/train/"
        )
        test_dir = (
            str(Path(self.data_root) / "val")
            if self.data_root
            else "data/datasets/CIL/domainnet/val/"
        )
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        if train_dset.class_to_idx != test_dset.class_to_idx:
            raise ValueError("Training and evaluation class directories must match")
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

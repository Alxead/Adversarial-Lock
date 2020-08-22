# -*-coding:utf-8-*-
import os
import math
import shutil
import logging
import numpy as np
import random
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] - [%(filename)s line:%(lineno)d] : %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_augmentation(config, is_train=True):
    aug = []
    if is_train:
        # random crop
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size, padding=4))
        # horizontal filp
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())

    aug.append(transforms.ToTensor())

    # if config.augmentation.normalize:
    #     if config.dataset == 'cifar10':
    #         aug.append(transforms.Normalize(
    #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    #     else:
    #         aug.append(transforms.Normalize(
    #             (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))

    if is_train and config.augmentation.cutout:
        # cutout
        aug.append(Cutout(n_holes=config.augmentation.holes,
                          length=config.augmentation.length))
    return aug


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if optimizer is not None:
            best_prec = checkpoint['best_prec']
            last_epoch = checkpoint['last_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=== done. also loaded optimizer from " +
                         "checkpoint '{}' (epoch {}) ===".format(
                             path, last_epoch + 1))
            return best_prec, last_epoch

def save_image(filename, data):
    data = data.clone().detach().cpu()

    std = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    mean = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))
    img = data.numpy()
    img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0]
        x = x[index]

        # x = Image.fromarray(x)

        if self.transform:
            x = self.transform(transforms.ToPILImage()(x))


        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def get_data_loader(transform_train, transform_test, config):

    orig_train_set = torchvision.datasets.CIFAR10(
        root=config.orig_data_path, train=True,
        download=True, transform=transform_train)

    orig_test_set = torchvision.datasets.CIFAR10(
        root=config.orig_data_path, train=False,
        download=True, transform=transform_test)

    encrypt_train_data = torch.cat(torch.load(config.encrypted_train_image))
    encrypt_train_labels = torch.cat(torch.load(config.encrypted_train_label))
    encrypt_train_set = CustomTensorDataset(tensors=(encrypt_train_data, encrypt_train_labels),
                                            transform=transform_train)

    encrypt_test_data = torch.cat(torch.load(config.encrypted_test_image))
    encrypt_test_labels = torch.cat(torch.load(config.encrypted_test_label))
    encrypt_test_set = CustomTensorDataset(tensors=(encrypt_test_data, encrypt_test_labels),
                                           transform=transform_test)


    #### loader

    loaders = {
        'orig_train': torch.utils.data.DataLoader(
            orig_train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=True),
        'orig_test': torch.utils.data.DataLoader(
            orig_test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=True),
        'enc_train': torch.utils.data.DataLoader(
            encrypt_train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=True),
        'enc_test': torch.utils.data.DataLoader(
            encrypt_test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=True),
    }

    return loaders


def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, config):
    lr = get_current_lr(optimizer)
    if config.lr_scheduler.type == 'STEP':
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == 'COSINE':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
            (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
            (1.0 + math.cos(math.pi * ratio)) / 2.0
    elif config.lr_scheduler.type == 'HTD':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
            (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
            (1.0 - math.tanh(
                config.lr_scheduler.lower_bound
                + (config.lr_scheduler.upper_bound
                   - config.lr_scheduler.lower_bound)
                * ratio)
             ) / 2.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

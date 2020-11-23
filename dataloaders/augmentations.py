# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
import math
import numbers
import random
import torch
import numpy as np
import cv2 as cv
from PIL import Image
import copy
import random
import SimpleITK as sitk


class CustomCompose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image_list):
        for method in self.augmentations:
            image_list = method(image_list)

        return image_list


# Flip / Rotate / Gaussian Noise
class RandomFlip(object):
    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, image_list):
        do_flip = np.random.random(1)
        if do_flip > 0.5:
            for i in range(len(image_list)):
                image_list[i] = np.flip(image_list[i], axis=self.axis)
        return image_list


class RandomRotate(object):
    def __init__(self):
        self.angle_list = [-10, -5, 0, 5, 10]

    def __call__(self, image_list):
        angle = self.angle_list[random.randint(0, 4)]

        img_slice = image_list[0]
        raws, z, cols = img_slice.shape
        M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (raws - 1) / 2.0), angle, 1)

        for i in range(len(image_list)):
            # print("rotate", key, i)
            for j in range(z):
                image_list[i][:, j, :] = self.rotate(image_list[i][:, j, :], M, raws, cols, is_target=(i==3))

        return image_list

    def rotate(self, img_slice, M, raws, cols, is_target=False):
        if not is_target:
            img_rotated = cv.warpAffine(img_slice, M, (cols, raws))
        else:
            img_rotated = cv.warpAffine(img_slice, M, (cols, raws), flags=cv.INTER_NEAREST)

        return img_rotated


class AddGaussianNoise(object):
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, image_list):
        # Check if a single image or a list of images has been passed
        if not isinstance(image_list, list):
            raise ValueError()

        # do not add on label
        offsets = np.random.normal(0, self.sigma, ([1] * (image_list[0].ndim - 1) + [image_list[0].shape[-1]]))
        for i in range(len(image_list) - 1):
            image_list[i] += offsets

        return image_list


# Crop
class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, list) or isinstance(size, tuple):
            self.size_x, self.size_y, self.size_z = size
        else:
            raise ValueError

    def __call__(self, image_list):
        x, y, z = image_list[0].shape

        if x < self.size_x or y < self.size_y and z < self.size_z:
            raise ValueError

        x1 = random.randint(0, x - self.size_x)
        y1 = random.randint(0, y - self.size_y)
        z1 = random.randint(0, z - self.size_z)

        for i in range(len(image_list)):
            image_list[i] = image_list[i][x1: x1 + self.size_x, y1: y1 + self.size_y, z1: z1 + self.size_z]

        return image_list


class CenterCrop(object):
    def __init__(self, size_ratio):
        if isinstance(size_ratio, list) or isinstance(size_ratio, tuple):
            self.size_ratio_x, self.size_ratio_y, self.size_ratio_z = size_ratio
        else:
            raise ValueError

    def __call__(self, image_list):
        x, y, z = image_list[0].shape
        self.size_x = self.size_ratio_x
        self.size_y = self.size_ratio_y
        self.size_z = self.size_ratio_z

        if x < self.size_x or y < self.size_y and z < self.size_z:
            raise ValueError

        x1 = int((x - self.size_x) / 2)
        y1 = int((y - self.size_y) / 2)
        z1 = int((z - self.size_z) / 2)

        for i in range(len(image_list)):
            image_list[i] = image_list[i][x1: x1 + self.size_x, y1: y1 + self.size_y, z1: z1 + self.size_z]

        return image_list


# To Tensor
class ToFloatTensor(object):
    def __init__(self, img_mean=0.5, img_std=0.5):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, image_list):
        for i in range(len(image_list)):
            tmp_img = image_list[i].astype(np.float32)
            image_list[i] = torch.unsqueeze(torch.from_numpy(tmp_img.copy()).float(), 0)

            if not _is_tensor_image(image_list[i]):
                raise TypeError('tensor is not a torch image.')

        return image_list


def _is_tensor_image(img):
    return torch.is_tensor(img)
    # and img.ndimension() == 3

import torch.utils.data as data
from PIL import Image
from skimage import io
from dataloaders.augmentations import CustomCompose, RandomFlip, RandomRotate, CenterCrop, RandomCrop
from dataloaders.augmentations import AddGaussianNoise, ToFloatTensor
import torchvision.transforms as transforms
import torch


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform_contrastive_learning(img_size=(72, 96, 72)):
    transform_list = []
    transform_list += [RandomFlip()]
    transform_list += [RandomRotate()]
    transform_list += [RandomCrop(img_size)]
    transform_list += [ToFloatTensor()]

    return CustomCompose(transform_list)


def get_transform_contrastive_learning_test(img_size=(72, 96, 72)):
    transform_list = [CenterCrop(img_size)]
    transform_list += [ToFloatTensor()]

    return CustomCompose(transform_list)
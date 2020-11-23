from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from torch import is_tensor
from torch.autograd import Variable


# Converts a Tensor into a float
def tensor2float(input_error):
    if is_tensor(input_error):
        error = input_error[0]
    elif isinstance(input_error, Variable):
        error = input_error.data[0]
    else:
        error = input_error
    return error


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if is_tensor(input_image):
        image_tensor = input_image
    elif isinstance(input_image, Variable):
        image_tensor = input_image.data
    else:
        return input_image

    image_numpy = image_tensor.data.cpu().float().numpy()
    # print(1, image_numpy.shape)
    img_2d = image_numpy[0, 0, :, 96, :]
    # print(1, img_2d.shape)
    image_numpy = np.tile(img_2d, (3, 1, 1))
    # print(2, image_numpy.shape)

    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy -= np.min(image_numpy)
    image_numpy /= np.max(image_numpy)
    image_numpy *= 255
    image_numpy.astype(np.uint8)

    # print(3, image_numpy.shape)
    return image_numpy

    # return input_image.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

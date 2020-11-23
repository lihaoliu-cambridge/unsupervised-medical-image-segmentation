# !/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import cv2

__author__ = "LIU Lihao"


# handle image value range
def transpose_image(img):
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    return img


def rescale_image_to_0_1(img):
    print(type(img))
    assert(isinstance(img, np.ndarray))

    img = img - np.min(img)
    img = img / np.max(img)

    img = transpose_image(img)

    return img


# print image min and max value
def print_min_max(img_arrays_list):
    for i in range(len(img_arrays_list)):
        assert (isinstance(img_arrays_list[i], np.ndarray))

        print("{}'s min and max:".format(i), np.min(img_arrays_list[i]), np.max(img_arrays_list[i]))


def print_edge_and_middle_value(img):
    assert (isinstance(img, np.ndarray))

    w, h = img.shape
    print(img[0:7, 0:7], img[int(w / 2) - 3:int(w / 2) + 3, int(h / 2) - 3:int(h / 2) + 3])


# plot image
def plot_image(img_arrays_list, image_names_list=None, save_path=None):
    plt.figure(figsize=(4*len(img_arrays_list), 4))
    plt.title(u"Images")

    for i in range(len(img_arrays_list)):
        plt.subplot(1, len(img_arrays_list), i+1)

        if not image_names_list or not image_names_list[i]:
            plt.title("Image_{}".format(i))
        else:
            plt.title(image_names_list[i])

        rescaled_image = rescale_image_to_0_1(img_arrays_list[i])

        if len(rescaled_image.shape) == 2:
            plt.imshow(rescaled_image, cmap="gray")
        else:
            plt.imshow(rescaled_image)
        plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=100)
    else:
        plt.show()


# generate mask and masked image
def make_heatmap_mask_from_mask(img_mask):
    assert (isinstance(img_mask, np.ndarray))

    img_mask = cv2.resize(np.uint8(255 * rescale_image_to_0_1(img_mask)), (256, 256))
    heatmap_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)

    return heatmap_mask


def make_masked_image(img, mask):
    assert (isinstance(img, np.ndarray) and isinstance(mask, np.ndarray))

    img = rescale_image_to_0_1(img)
    result = mask * 0.3 + np.uint8(255 * img) * 0.5
    cv2.imwrite("./datasets/tmp/result.jpg", result)

    return result

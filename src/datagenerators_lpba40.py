# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk



def miccai2018_gen_lpba40(gen, atlas_vol_bs, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X, Y = next(gen)[0]
        if bidir:
            yield ([X, Y], [Y, X, zeros])
        else:
            yield ([X, Y], [Y, zeros,
                            np.zeros((batch_size, 80, 96, 112, 6)), np.zeros((batch_size, 40, 48, 56, 6)), np.zeros((batch_size, 80, 96, 112, 3)),
                            np.zeros((batch_size, 80, 96, 112, 64)), np.zeros((batch_size, 40, 48, 56, 128)), np.zeros((batch_size, 20, 24, 28, 256)),
                            ])


def example_gen_lpba40(vol_names, batch_size=1, return_segs=False, image_path="/data/lpba40/Brains_MNIspace_reglinear", seg_dir=None):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """
    # todo:


    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        Y_data = []
        for idx in idxes:
            X = load_volfile_lpba40_vol(vol_names[idx], fixed=True)
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

            fixed_name = vol_names[idx].strip().split("/")[-1].split(".")[-2].split("_")[-1]
            fixed_path = os.path.join(image_path, "{}_to_{}.hdr".format(str(fixed_name), str(fixed_name)))
            Y = load_volfile_lpba40_vol(fixed_path, fixed=True)
            Y = Y[np.newaxis, ..., np.newaxis]
            Y_data.append(Y)

        if batch_size > 1:
            return_vals = [[np.concatenate(X_data, 0), np.concatenate(Y_data, 0)]]
        else:
            return_vals = [[X_data[0], Y_data[0]]]

        # also return segmentations
        if return_segs:
            X_data = []
            for idx in idxes:
                X_seg = load_volfile_lpba40(vol_names[idx])

                X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_data.append(X_seg)

                fixed_name = vol_names[idx].strip().split("/")[-1].split(".")[-2].split("_")[-1]
                fixed_path = os.path.join(image_path, "{}_to_{}.hdr".format(str(fixed_name), str(fixed_name)))
                Y_seg = load_volfile_lpba40(fixed_path)
                Y_seg = Y_seg[np.newaxis, ..., np.newaxis]
                Y_data.append(Y_seg)

            if batch_size > 1:
                return_vals.append([np.concatenate(X_data, 0), np.concatenate(Y_data, 0) ])
            else:
                return_vals.append([X_data[0], Y_data[0]])

        yield tuple(return_vals)


def load_example_by_name(vol_name, seg_name, fixed=False):
    """
    load a specific volume and segmentation
    """
    X = load_volfile_lpba40_vol(vol_name, fixed=fixed)
    X = X[np.newaxis, ..., np.newaxis]
    return_vals = [X]

    X_seg = load_volfile_lpba40(seg_name)
    X_seg = X_seg[np.newaxis, ..., np.newaxis]
    X_seg[X_seg == 181] = 0
    X_seg[X_seg == 182] = 0
    return_vals.append(X_seg)

    return tuple(return_vals)


def correct_box(box, image_size):
    if box[0] < 0:
        box[0] = 0
        box[1] = 144

    if box[1] > image_size[0]:
        box[0] = image_size[0] - 144
        box[1] = image_size[0]

    if box[2] < 0:
        box[2] = 0
        box[3] = 176

    if box[3] > image_size[1]:
        box[2] = image_size[1] - 176
        box[3] = image_size[1]

    if box[4] < 0:
        box[4] = 0
        box[5] = 144

    if box[5] > image_size[2]:
        box[4] = image_size[2] - 144
        box[5] = image_size[2]

    return box


def load_volfile_lpba40_vol(datafile, box=(0, 0, 0), fixed=False):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data'
    """
    X = sitk.GetArrayFromImage(sitk.ReadImage(str(datafile))).swapaxes(0, 2)

    if box == (0, 0, 0):
        if fixed:
            return std_0_to_1(whitening(X))
        else:
            return whitening(X)
    else:
        a, b, c, d, e, f = correct_box([int(box[0]) - 72,
                                        int(box[0]) + 72,
                                        int(box[1]) - 88,
                                        int(box[1]) + 88,
                                        int(box[2]) - 72,
                                        int(box[2]) + 72],
                                       X.shape)
        if fixed:
            return std_0_to_1(whitening(X[a:b, c:d, e:f]))
        else:
            return whitening(X[a:b, c:d, e:f])


def load_volfile_lpba40(datafile, box=(0, 0, 0)):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data'
    """
    X = sitk.GetArrayFromImage(sitk.ReadImage(str(datafile))).swapaxes(0, 2)

    if box == (0, 0, 0):
        return X
    else:
        a, b, c, d, e, f = correct_box([int(box[0]) - 72,
                                        int(box[0]) + 72,
                                        int(box[1]) - 88,
                                        int(box[1]) + 88,
                                        int(box[2]) - 72,
                                        int(box[2]) + 72],
                                       X.shape)
        return X[a:b, c:d, e:f]


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""
    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.

    return ret


def std_0_to_1(image):
    image -= np.min(image)
    image /= np.max(image)

    return image

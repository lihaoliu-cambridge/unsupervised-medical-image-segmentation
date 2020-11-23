from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np


def dice(vol1, vol2, num_classes=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    num_classes : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''

    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    dicem = np.zeros(len(num_classes))
    dicem2 = np.zeros(len(num_classes))
    for idx, lab in enumerate(num_classes):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2. * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = 1.0 * top / bottom
        dicem2[idx] = 1.0 * int(np.sum(vol1l)) / (vol1l.shape[2] * vol1l.shape[3] * vol1l.shape[4])

    if nargout == 1:
        return dicem, dicem2
    else:
        return (dicem, dicem2, num_classes)


def abs_vol_difference(predictions, labels, num_classes):
    """Calculates the absolute volume difference for each class between
        labels and predictions.

    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate avd for

    Returns:
        np.ndarray: avd per class
    """

    avd = np.zeros((num_classes))
    eps = 1e-6
    for i in range(num_classes):
        avd[i] = np.abs(np.sum(predictions == i) - np.sum(labels == i)
                        ) / (np.float(np.sum(labels == i)) + eps)

    return avd.astype(np.float32)


def crossentropy(predictions, labels, logits=True):
    """Calculates the crossentropy loss between predictions and labels

    Args:
        prediction (np.ndarray): predictions
        labels (np.ndarray): labels
        logits (bool): flag whether predictions are logits or probabilities

    Returns:
        float: crossentropy error
    """

    if logits:
        maxes = np.amax(predictions, axis=-1, keepdims=True)
        softexp = np.exp(predictions - maxes)
        softm = softexp / np.sum(softexp, axis=-1, keepdims=True)
    else:
        softm = predictions
    loss = np.mean(-1. * np.sum(labels * np.log(softm + 1e-8), axis=-1))
    return loss.astype(np.float32)

# -*- coding: utf-8 -*-
import os
import sys

import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session

# project
sys.path.append('../ext/medipy-lib')
# import util
from medipy.metrics import dice
from src import datagenerators_lpba40 as datagenerators_lpba40, networks_lpba40 as networks_lpba40

# Test file and anatomical labels we want to evaluate
test_brain_file = open('../data/test_pairs.txt')
test_brain_strings = test_brain_file.readlines()
test_brain_strings = [x.strip() for x in test_brain_strings]
n_batches = len(test_brain_strings)
good_labels = [  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63,
                 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                 90, 91, 92, 101, 102, 121, 122, 161, 162, 163, 164, 165, 166]
good_labels_FL = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
good_labels_PL = [41, 42, 43, 44, 45]
good_labels_OL = [61, 62, 63, 64, 65, 66, 67, 68]
good_labels_TL = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]
good_labels_CL = [101, 102, 121, 122]
good_labels_Ptm = [163, 164]
good_labels_Hpcp = [165, 166]
# no 181, 182
# print("labels", len(good_labels), good_labels)


def test(gpu_id=0,
         model_dir="../model/lpba40",
         iter_num="00",
         compute_type = 'GPU',  # GPU or CPU
         vol_size=(160, 192, 224),
         nf_enc=[16,32,32,32],
         nf_dec=[32,32,32,32,16,3],
         save_file=None):
    """
    test via segmetnation propagation
    works by iterating over some iamge files, registering them to atlas,
    propagating the warps, then computing Dice with atlas segmentations
    """

    # GPU handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        # if testing miccai run, should be xy indexing.
        net = networks_lpba40.miccai2018_net(vol_size, nf_enc, nf_dec, use_miccai_int=False, indexing='ij')
        net.load_weights(os.path.join(model_dir, str(iter_num) + '.h5'))
        print(os.path.join(model_dir, str(iter_num) + '.h5'))

        # compose diffeomorphic flow output model
        diff_net = keras.models.Model(net.inputs, net.get_layer('diffflow').output)

        # NN transfer model
        nn_trf_model = networks_lpba40.nn_trf(vol_size, indexing='ij')
        nn_trf_model = networks_lpba40.nn_trf(vol_size, indexing='ij')

    # if CPU, prepare grid
    if compute_type == 'CPU':
        # grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)
        print('Error: No GPU.')

    # prepare a matrix of dice values
    dice_vals = np.zeros((len(good_labels), n_batches))
    dice_vals_FL = np.zeros((len(good_labels_FL), n_batches))
    dice_vals_PL = np.zeros((len(good_labels_PL), n_batches))
    dice_vals_OL = np.zeros((len(good_labels_OL), n_batches))
    dice_vals_TL = np.zeros((len(good_labels_TL), n_batches))
    dice_vals_CL = np.zeros((len(good_labels_CL), n_batches))
    dice_vals_Ptm = np.zeros((len(good_labels_Ptm), n_batches))
    dice_vals_Hpcp = np.zeros((len(good_labels_Hpcp), n_batches))

    for k in range(n_batches):
        print(111)
        # get data
        vol_name, atlas_vol_name = test_brain_strings[k].split(",")

        # seg
        seg_name = vol_name.replace("/data/lpba40/Brains_MNIspace_reglinear/",
                                    "/data/lpba40/Segmentations/")
        atlas_seg_name = atlas_vol_name.replace("/data/lpba40/Brains_MNIspace_reglinear/",
                                    "/data/lpba40/Segmentations/")
        # vol
        X_vol, X_seg = datagenerators_lpba40.load_example_by_name(vol_name, seg_name, fixed=True)
        atlas_vol, atlas_seg = datagenerators_lpba40.load_example_by_name(atlas_vol_name, atlas_seg_name, fixed=True)
        atlas_seg = atlas_seg[0, ..., 0]

        # predict transform
        with tf.device(gpu):
            pred = diff_net.predict([X_vol, atlas_vol])
            [y, flow_params, flow_params0, flow_params1, flow_params2, y0, y1, y2] = net.predict([X_vol, atlas_vol])

        # Warp segments with flow
        if compute_type == 'CPU':
            print('Error: No GPU.')
        else:
            warp_seg = nn_trf_model.predict([X_seg, pred])[0,...,0]

        # compute Volume Overlap (Dice)
        dice_vals[:, k] = dice(warp_seg, atlas_seg, labels=good_labels)
        dice_vals_FL[:, k] = dice(warp_seg, atlas_seg, labels=good_labels_FL)
        dice_vals_PL[:, k] = dice(warp_seg, atlas_seg, labels=good_labels_PL)
        dice_vals_OL[:, k] = dice(warp_seg, atlas_seg, labels=good_labels_OL)
        dice_vals_TL[:, k] = dice(warp_seg, atlas_seg, labels=good_labels_TL)
        dice_vals_CL[:, k] = dice(warp_seg, atlas_seg, labels=good_labels_CL)
        dice_vals_Ptm[:, k] = dice(warp_seg, atlas_seg, labels=good_labels_Ptm)
        dice_vals_Hpcp[:, k] = dice(warp_seg, atlas_seg, labels=good_labels_Hpcp)

        print('%s %3d: %5.3f All: %5.3f' % (vol_name, k, np.mean(dice_vals[:, k]), np.mean(np.mean(dice_vals[:, :k+1]))))
        print('%s %3d: %5.3f All: %5.3f' % ("FL", k, np.mean(dice_vals_FL[:, k]), np.mean(np.mean(dice_vals_FL[:, :k+1]))))
        print('%s %3d: %5.3f All: %5.3f' % ("PL", k, np.mean(dice_vals_PL[:, k]), np.mean(np.mean(dice_vals_PL[:, :k+1]))))
        print('%s %3d: %5.3f All: %5.3f' % ("OL", k, np.mean(dice_vals_OL[:, k]), np.mean(np.mean(dice_vals_OL[:, :k+1]))))
        print('%s %3d: %5.3f All: %5.3f' % ("TL", k, np.mean(dice_vals_TL[:, k]), np.mean(np.mean(dice_vals_TL[:, :k+1]))))
        print('%s %3d: %5.3f All: %5.3f' % ("CL", k, np.mean(dice_vals_CL[:, k]), np.mean(np.mean(dice_vals_CL[:, :k+1]))))
        print('%s %3d: %5.3f All: %5.3f' % ("Ptm", k, np.mean(dice_vals_Ptm[:, k]), np.mean(np.mean(dice_vals_Ptm[:, :k+1]))))
        print('%s %3d: %5.3f All: %5.3f' % ("Hpcp", k, np.mean(dice_vals_Hpcp[:, k]), np.mean(np.mean(dice_vals_Hpcp[:, :k+1]))))

    if save_file is not None:
        sio.savemat(save_file, {'dice_vals': dice_vals, 'labels': good_labels})


if __name__ == "__main__":
    """
    assuming the model is model_dir/iter_num.h5
    python test_lpba40.py gpu_id model_dir iter_num
    """
    test()

# -*- coding: utf-8 -*-
# py imports
import os
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model 

# project imports
import datagenerators_lpba40 as datagenerators_lpba40
import networks_lpba40 as networks_lpba40
import losses

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen


def train(data_dir,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          prior_lambda,
          image_sigma,
          steps_per_epoch,
          batch_size,
          load_model_file,
          bidir,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param model_dir: model folder to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param nb_epochs: number of training iterations
    :param prior_lambda: the prior_lambda, the scalar in front of the smoothing laplacian, in MICCAI paper
    :param image_sigma: the image sigma in MICCAI paper
    :param steps_per_epoch: frequency with which to save model
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param bidir: logical whether to use bidirectional cost function
    """

    vol_size = (160, 192, 224)

    # Fisrt 30 images are used as training set, and the rest 10 images are testing set.
    constrain = list(range(31, 41, 1))
    moving_path = []
    list_dirs = os.walk(data_dir)
    for root, dirs, files in list_dirs:
        for f in files:
            if f.endswith(".hdr") and f.startswith("l"):
                for c in constrain:
                    if "l{}_".format(str(c)) in str(f) or "_l{}.".format(str(c)) in str(f):
                        break
                else:
                    moving_path.append(os.path.join(data_dir, f))

    train_vol_names = moving_path
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, "Could not find any training data"

    nf_enc = [16,32,32,32]
    nf_dec = [32,32,32,32,16,3]

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # gpu handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):
        model = networks_lpba40.miccai2018_net(vol_size, nf_enc, nf_dec, bidir=bidir)

        # load initial weights
        if load_model_file is not None and load_model_file != "":
            print("load file from {}".format(load_model_file))
            model.load_weights(load_model_file)

        # save first iteration
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

        # compile
        flow_vol_shape = model.outputs[1].shape[1:-1]
        loss_class = losses.Miccai2018(image_sigma, prior_lambda, flow_vol_shape=flow_vol_shape)
        if bidir:
            model_losses = [loss_class.recon_loss, loss_class.recon_loss, loss_class.kl_loss]
            loss_weights = [0.5, 0.5, 1]
        else:
            model_losses = [loss_class.recon_loss, loss_class.kl_loss,
                            loss_class.kl_loss_1, loss_class.kl_loss_2, loss_class.kl_loss_3,
                            loss_class.mse_loss, loss_class.mse_loss, loss_class.mse_loss]
            loss_weights = [1, 1, 0.5, 0.5, 0.5, 1, 1, 1]


    # data generator
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    train_example_gen = datagenerators_lpba40.example_gen_lpba40(train_vol_names, batch_size=batch_size, image_path=data_dir)
    miccai2018_gen_lpba40 = datagenerators_lpba40.miccai2018_gen_lpba40(train_example_gen,
                                                                        vol_size,
                                                                        batch_size=batch_size,
                                                                        bidir=bidir)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')

    # fit generator
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)

        # single gpu
        else:
            save_callback = ModelCheckpoint(save_file_name)
            mg_model = model

        mg_model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
        mg_model.fit_generator(miccai2018_gen_lpba40,
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=[save_callback],
                               steps_per_epoch=steps_per_epoch,
                               verbose=1)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str,
                        help="data folder", default='../data/lpba40/Brains_MNIspace_reglinear/',)
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../model/lpba40/',
                        help="model folder")
    parser.add_argument("--gpu", type=str, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=5000,
                        help="number of iterations")
    parser.add_argument("--prior_lambda", type=float,
                        dest="prior_lambda", default=10,
                        help="prior_lambda regularization parameter")
    parser.add_argument("--image_sigma", type=float,
                        dest="image_sigma", default=0.02,
                        help="image noise parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=500,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='../model/miccai2018_10_02_init1.h5',
                        help="optional h5 model file to initialize with")
    parser.add_argument("--bidir", type=int,
                        dest="bidir", default=0,
                        help="whether to use bidirectional cost function")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="first epoch")

    args = parser.parse_args()
    train(**vars(args))

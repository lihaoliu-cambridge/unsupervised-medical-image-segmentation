# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, LeakyReLU, Deconvolution3D, PReLU
from keras.initializers import RandomNormal
import sys
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')

from keras.layers import Layer
import tensorflow as tf

K.set_image_data_format("channels_last")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def incept(inputs, num_channel, activation='linear'):
    z1 = Conv3D(num_channel, (2, 2, 2), strides=(2, 2, 2), padding='same', activation=activation)(inputs)
    z2 = Conv3D(num_channel, (5, 5, 5), strides=(2, 2, 2), padding='same', activation=activation)(inputs)
    z3 = Conv3D(num_channel, (7, 7, 7), strides=(2, 2, 2), padding='same', activation=activation)(inputs)
    z4 = Conv3D(num_channel, (11, 11, 11), strides=(2, 2, 2), padding='same', activation=activation)(inputs)

    z = concatenate([z4, z3, z2, z1])
    z = PReLU(shared_axes=[1, 2, 3])(z)

    return z


def unet_model_3d(input_shape,
                  pool_size=(2, 2, 2),
                  deconvolution=False,
                  depth=4,
                  n_base_filters=16,
                  batch_normalization=False):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    src = Input(shape=input_shape + (1,))
    tgt = Input(shape=input_shape + (1,))
    levels_src = list()
    levels_tgt = list()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ src ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    current_layer_src = incept(src, n_base_filters // 4)
    levels_src.append([current_layer_src])

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer_src, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        # print("l1", layer1.shape)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)

        # print("l2", layer1.shape)
        if layer_depth < depth - 1:
            current_layer_src = MaxPooling3D(pool_size=pool_size, data_format="channels_last")(layer2)
            levels_src.append([layer1, layer2, current_layer_src])
        else:
            current_layer_src = layer2
            levels_src.append([layer1, layer2])

        # print("c", current_layer_src.shape)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ middle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    levels_z = list()
    for layer_depth in range(depth):
        # print(layer_depth)
        input = levels_src[layer_depth][-1]

        z_mean = create_convolution_block(input_layer=input,
                                          n_filters=3,
                                          batch_normalization=batch_normalization,
                                          z="mean")
        z_log_var = create_convolution_block(input_layer=input,
                                             n_filters=3,
                                             batch_normalization=batch_normalization,
                                             z="var")

        z = Sample(name="z_layer_{}".format(str(layer_depth)))([z_mean, z_log_var])

        output_y = create_convolution_block(input_layer=z,
                                            n_filters=input._keras_shape[-1],
                                            batch_normalization=batch_normalization)

        if layer_depth == 0:
            up_z = z
        else:
            up_z = get_up_convolution(pool_size=(2 ** (layer_depth), 2 ** (layer_depth), 2 ** (layer_depth)),
                                      deconvolution=deconvolution,
                                      n_filters=z._keras_shape[-1])(z)

        levels_z.append([up_z, output_y, z_mean, z_log_var])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tgt ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    current_layer_tgt = incept(tgt, n_base_filters // 4)
    levels_tgt.append([current_layer_tgt])

    for layer_depth in range(depth):
        # print(levels_z[layer_depth][1].shape)
        current_layer_tgt = concatenate([levels_z[layer_depth][1], current_layer_tgt], axis=-1)
        # print("c", current_layer_tgt.shape)
        layer1 = create_convolution_block(input_layer=current_layer_tgt, n_filters=n_base_filters * (2 ** layer_depth),
                                          batch_normalization=batch_normalization)
        # print("l1", layer1.shape)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters * (2 ** layer_depth) * 2,
                                          batch_normalization=batch_normalization)

        if layer_depth < depth -1:
            current_layer_tgt = MaxPooling3D(pool_size=pool_size, data_format="channels_last")(layer2)
            levels_tgt.append([layer1, layer2, current_layer_tgt])
        else:
            current_layer_tgt = layer2
            levels_tgt.append([layer1, layer2])

    current_layer = concatenate([current_layer_tgt, current_layer_src], axis=-1)
    current_layer = create_convolution_block(input_layer=current_layer,
                                             strides=(1, 1, 1),
                                             n_filters=current_layer._keras_shape[-1] // 2,
                                             batch_normalization=batch_normalization)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ add levels with max pooling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[-1])(current_layer)
        concat = concatenate([up_convolution, levels_tgt[layer_depth][-1]], axis=-1)

        current_layer = create_convolution_block(n_filters=levels_tgt[layer_depth][-1]._keras_shape[-1],
                                                 input_layer=concat,
                                                 batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels_tgt[layer_depth][-1]._keras_shape[-1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(16, (1, 1, 1), data_format="channels_last")(current_layer)
    act = LeakyReLU(0.2)(final_convolution)

    flow_params0 = concatenate([levels_z[0][2], levels_z[0][3]])
    flow_params1 = concatenate([levels_z[1][2], levels_z[1][3]])
    flow_params2 = concatenate([levels_z[2][2], levels_z[2][3]])
    y0 = concatenate([levels_z[0][1], levels_tgt[0][-1]])
    y1 = concatenate([levels_z[1][1], levels_tgt[1][-1]])
    y2 = concatenate([levels_z[2][1], levels_tgt[2][-1]])

    return Model(inputs=[src, tgt], outputs=[act,
                                             levels_z[0][0], levels_z[1][0], levels_z[2][0],
                                             flow_params0, flow_params1, flow_params2,
                                             y0, y1, y2])


def create_convolution_block(input_layer,
                             n_filters,
                             batch_normalization=False,
                             kernel=(3, 3, 3),
                             activation=None,
                             padding='same',
                             strides=(1, 1, 1),
                             instance_normalization=False,
                             z=None):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    if z is None:
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, data_format="channels_last")(input_layer)
    elif z == "mean":
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, data_format="channels_last", kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(input_layer)
    elif z == "var":
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, data_format="channels_last", kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10))(input_layer)


    if batch_normalization:
        layer = BatchNormalization(axis=-1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=-1)(layer)

    if activation is None:
        return LeakyReLU(0.2)(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None] + output_image_shape + [n_filters])


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides, data_format="channels_last")
    else:
        return UpSampling3D(size=pool_size, data_format="channels_last")


# Helper functions
def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z


class Sample(Layer):
    """
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
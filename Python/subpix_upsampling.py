from __future__ import absolute_import
from __future__ import division

import pdb
import copy
import inspect
import warnings
import numpy as np
import tensorflow as tf
import types as python_types
from keras import backend as K
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D
from keras.engine import InputSpec, Layer, Merge
from keras import regularizers
from keras.utils.np_utils import conv_output_length
from keras.utils.generic_utils import func_dump, func_load
from keras.layers.recurrent import Recurrent, time_distributed_dense
from keras import activations, initializations, regularizers, constraints
from keras.layers import BatchNormalization
from keras.layers import (
     Input,
     Activation,
     Merge,
     merge,
     Dropout,
     Reshape,
     Permute,
     Dense,
     UpSampling2D,
     Flatten,
     Lambda
    )


# ===================================
#  Subpixel Dense Up-sampling Layer.
# ===================================

def dense_interp(x, r, shape):
    # x should be of shape : batch_size, w, h, r^2
    bsize, a, b, c = shape
    X = tf.reshape(x, (-1, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    X = tf.reshape(X, (-1, b, a * r, r))
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (-1, a * r, b * r, 1))


class Subpix_denseUP(Layer):
    '''
    This layer is inspired by the paper[1],
    aiming to provide upsampling in a more accurate way.

    Input to this layer is of size: w x h x r^2*c 
    
    Output from this layer is of size : W x H x C
    where W = w * r, H = h * r, C = c

    Intuitively, we want to compensate the information loss with more feature 
    channels, depth -> spatial resolution.
    We can reshape the feature channels, 
    for example, take 1 x 1 x k^2, we can reshape it to k x k x 1.

    ratio : the ratio you want to upsample for both dimensions (w,h).
    nb_channel : The channels you really want after up-sampling.
    nb_channel = input_shape[-1] / (prod(ratio))
    dim_ordering = 'tf' 

    Reference:
    [1] Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network.
    '''

    def __init__(self, ratio=2, dim_ordering='tf', **kwargs):
        self.ratio = ratio
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]
        super(Subpix_denseUP, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'tf':
            height = self.ratio * input_shape[1] if input_shape[1] is not None else None
            width = self.ratio * input_shape[2] if input_shape[2] is not None else None
            if input_shape[3] % (self.ratio ** 2) != 0:
                raise Exception('input shape : {}, can not upsample to {}'.format(input_shape,
                                                                                  (input_shape[0], height, width,
                                                                                   input_shape[3] // self.ratio ** 2)))
            else:
                channel = input_shape[3] // (self.ratio ** 2)
            return (input_shape[0],
                    width,
                    height,
                    channel)
        else:
            raise Exception('Only support TF, Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        inp_shape = x._keras_shape
        output_shape = self.get_output_shape_for(inp_shape)
        nb_channel = output_shape[-1]
        r = self.ratio

        if nb_channel > 1:
            interp_shape = [inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3] // nb_channel]
            Xc = tf.split(x, nb_channel, 3)
            X = tf.concat([dense_interp(x, r, interp_shape) for x in Xc], 3)
        else:
            interp_shape = inp_shape
            X = dense_interp(x, r, interp_shape)
        return X

    def get_config(self):
        config = {'ratio': self.ratio}
        base_config = super(Subpix_denseUP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

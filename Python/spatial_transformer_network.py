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

# =============================================================================
#                     Spatial Transformer Layer
# =============================================================================


def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])


def _interpolate(im, x, y, downsample_factor):
    # constants
    num_batch = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    num_channels = tf.shape(im)[3]

    # need to convert to float.
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    out_height = tf.cast(height_f // downsample_factor, 'int32')
    out_width = tf.cast(width_f // downsample_factor, 'int32')

    zero = tf.zeros([], dtype='int32')
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_x = tf.cast(tf.shape(im)[1] - 1, 'int32')

    # scale indices from [-1, 1] to [0, width or height]
    x = (x + 1.0) * (width_f) / 2.0
    y = (y + 1.0) * (height_f) / 2.0

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clips tensor values to a specified min and max. (outside of the boundary)
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    dim2 = width
    dim1 = width * height
    base = _repeat(tf.range(num_batch, dtype='int32') * dim1, out_height * out_width)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore channels dim
    im_flat = tf.reshape(im, tf.stack([-1, num_channels]))
    im_flat = tf.cast(im_flat, 'float32')
    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)  # ratio
    wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
    wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
    wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
    output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return output
    constants


def _meshgrid(height, width):
    x_t_flat, y_t_flat = tf.meshgrid(tf.linspace(-1., 1., width), tf.linspace(-1., 1., height))
    ones = tf.ones_like(x_t_flat)
    grid = tf.concat(values=[x_t_flat, y_t_flat, ones], axis=0)
    return grid


def _transform(theta, conv_input, downsample_factor):
    # theta : transformation matrix of size batch_sz x 6
    # input : the feature maps that to be transformed.
    # in this code, I only wrote the tenforflow backend.
    num_batch = tf.shape(conv_input)[0]
    height = tf.shape(conv_input)[1]
    width = tf.shape(conv_input)[2]
    num_channels = tf.shape(conv_input)[3]

    theta = tf.reshape(theta, (-1, 2, 3))  # Reshape to batch_sz x 2 x 3
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')

    out_height = tf.cast(height_f // downsample_factor, 'int32')
    out_width = tf.cast(width_f // downsample_factor, 'int32')

    # Generate the Target Grid Coordinates
    grid = _meshgrid(out_height, out_width)
    grid = tf.expand_dims(grid, 0)  # 1 x w x h x 3
    grid = tf.reshape(grid, [-1])  # flatten to into 1-D (wh3,)
    grid = tf.tile(grid, tf.stack([num_batch]))
    grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

    # Transform the target grid back to the source grid, where original image lies.
    T_g = tf.matmul(theta, grid)
    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
    x_s_flat = tf.reshape(x_s, [-1])
    y_s_flat = tf.reshape(y_s, [-1])

    # input : (bs, height, width, channels)
    input_transformed = _interpolate(conv_input, x_s_flat, y_s_flat, downsample_factor)

    output = tf.reshape(input_transformed,
                        (num_batch, out_height, out_width, num_channels))
    return output


class STN(Layer):
    '''
    Spatial Transformer Layer
    This file is highly based on [1]_, written by taoyizhi68.
    Though original code was based on Theano.
    Implements a spatial transformer layer as described in [2]_.
    Parameters
    ------------------------------------------------------------
    inputs : a list of [:class:`Layer` instance or a tuple]
    The layers feeding into this layer. 
    
    The list must have two entries with the first network being a 
    convolutional net and the second layer being the transformation matrices. 
    The first network should have output shape [batch_size, height, width, num_channels]. (tf mode)
    The output of the second network should be [batch_size, 6]. (Affine Transformation)

    downsample_factor : float
    A value of 1 will keep the original size of the image.
    Values larger than 1 will down sample the image. 
    Values below 1 will upsample the image.
    example image: height= 100, width = 200
    downsample_factor = 2
    output image will then be 50, 100
    ----------
    References
    ----------
    .. [1]  https://github.com/taoyizhi68/keras-Spatial-Transformer-Layer/blob/master/SpatialTransformer.py
    .. [2]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    '''

    def __init__(self, downsample_factor=1, **kwargs):
        super(STN, self).__init__(**kwargs)
        self.downsample_factor = downsample_factor

    def get_output_shape_for(self, input_shape):
        # input dims are (bs, height, width, num_filters)
        # Scale height and width by downsample factor
        rows = input_shape[0][1]
        cols = input_shape[0][2]
        return (input_shape[0][0], rows // self.downsample_factor,
                cols // self.downsample_factor, input_shape[0][-1])

    def call(self, x, mask=None):
        # theta should be shape (batchsize, 6)
        # see eq. (1) and sec 3.1 in ref [2]
        # conv_input is the output feature maps that need to be transformed.
        conv_input, theta = x
        output = _transform(theta, conv_input, self.downsample_factor)
        return output

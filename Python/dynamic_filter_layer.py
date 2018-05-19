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
#                     Dynamic Filter Networks
# =============================================================================


class Dynamic_Filter_Layer(Layer):
    '''
    This layer needs to take two inputs(layers),
    just like the STN.
    
    Only support border_mode = 'valid', 
    if you want the output to have same size as input please do padding 
    before feeding to this layer. 
    '''

    def __init__(self, ksize=5, dim_ordering='tf',
                 border_mode='valid',
                 activation=None,
                 subsample=(1, 1), **kwargs):
        self.kernel_size = ksize
        self.subsample = subsample
        self.border_mode = border_mode
        self.dim_ordering = dim_ordering
        self.activation = activations.get(activation)
        super(Dynamic_Filter_Layer, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'tf':
            rows = input_shape[0][1]
            cols = input_shape[0][2]
        else:
            raise ValueError('Only support tensorflow.')

        rows = conv_output_length(rows, self.kernel_size,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.kernel_size,
                                  self.border_mode, self.subsample[1])

        return (input_shape[0][0], rows, cols, input_shape[0][-1])

    def call(self, x, mask=None):
        input_, flow_layer_ = x
        stride_row, stride_col = self.subsample
        shape = input_._keras_shape
        output_row = shape[1] - self.kernel_size + 1
        output_col = shape[2] - self.kernel_size + 1
        xs = []
        ws = []
        for i in range(output_row):
            for j in range(output_col):
                slice_row = slice(i * stride_row,
                                  i * stride_row + self.kernel_size)
                slice_col = slice(j * stride_col,
                                  j * stride_col + self.kernel_size)
                xs.append(K.reshape(input_[:, slice_row, slice_col, :],
                                    (1, -1, self.kernel_size ** 2, shape[-1])))
                ws.append(K.reshape(flow_layer_[:, i, j, :], (1, -1, self.kernel_size ** 2, 1)))
        x_aggregate = K.concatenate(xs, axis=0)
        x_aggregate = K.permute_dimensions(x_aggregate, (0, 1, 3, 2))
        W = K.concatenate(ws, axis=0)
        output = K.batch_dot(x_aggregate, W)
        output = K.reshape(output, (output_row, output_col, -1, shape[3]))
        output = K.permute_dimensions(output, (2, 0, 1, 3))
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'ksize': self.kernel_size,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering}
        base_config = super(Dynamic_Filter_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

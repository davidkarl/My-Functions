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

# ===================================================
#                Correlation Layer
# ===================================================


class Normalized_Correlation_Layer(Layer):
    '''
    This layer does Normalized Correlation.
    
    It needs to take two inputs(layers),
    currently, it only supports the border_mode = 'valid',
    if you need to output the same shape as input, 
    do padding before giving the layer.
    
    '''

    def __init__(self, patch_size=5,
                 dim_ordering='tf',
                 border_mode='valid',
                 stride=(2, 2),
                 activation=None,
                 **kwargs):
        if border_mode != 'valid':
            raise ValueError('Invalid border mode for Correlation Layer '
                             '(only "valid" is supported):', border_mode)
        self.kernel_size = patch_size
        self.subsample = stride
        self.dim_ordering = dim_ordering
        self.border_mode = border_mode
        self.activation = activations.get(activation)
        super(Normalized_Correlation_Layer, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'tf':
            inp_rows = input_shape[0][1]
            inp_cols = input_shape[0][2]
        else:
            raise ValueError('Only support tensorflow.')
        rows = conv_output_length(inp_rows, self.kernel_size,
                                   self.border_mode, 1)
        cols = conv_output_length(inp_cols, self.kernel_size,
                                   self.border_mode, 1)
        out_r = conv_output_length(inp_rows, self.kernel_size,
                                   self.border_mode, self.subsample[0])
        out_c = conv_output_length(inp_cols, self.kernel_size,
                                   self.border_mode, self.subsample[1])
        return (input_shape[0][0], rows, cols, out_r * out_c)

    def call(self, x, mask=None):
        input_1, input_2 = x
        stride_row, stride_col = self.subsample
        inp_shape = input_1._keras_shape
        output_shape = self.get_output_shape_for([inp_shape, inp_shape])
        output_row = inp_shape[1] - self.kernel_size + 1
        output_col = inp_shape[2] - self.kernel_size + 1
        xc_1 = []
        xc_2 = []
        for i in range(output_row):
            for j in range(output_col):
                slice_row = slice(i, i + self.kernel_size)
                slice_col = slice(j, j + self.kernel_size)
                xc_2.append(K.reshape(input_2[:, slice_row, slice_col, :],
                                      (-1, 1, self.kernel_size**2*inp_shape[-1])))
                if i % stride_row == 0 and j % stride_col == 0:
                    xc_1.append(K.reshape(input_1[:, slice_row, slice_col, :],
                                          (-1, 1, self.kernel_size**2*inp_shape[-1])))

        xc_1_aggregate = K.concatenate(xc_1, axis=1) # batch_size x w'h' x (k**2*d), w': w/subsample-1
        xc_1_mean = K.mean(xc_1_aggregate, axis=-1, keepdims=True)
        xc_1_std = K.std(xc_1_aggregate, axis=-1, keepdims=True)
        xc_1_aggregate = (xc_1_aggregate - xc_1_mean) / xc_1_std

        xc_2_aggregate = K.concatenate(xc_2, axis=1) # batch_size x wh x (k**2*d), w: output_row
        xc_2_mean = K.mean(xc_2_aggregate, axis=-1, keepdims=True)
        xc_2_std = K.std(xc_2_aggregate, axis=-1, keepdims=True)
        xc_2_aggregate = (xc_2_aggregate - xc_2_mean) / xc_2_std
        
        
        xc_1_aggregate = K.permute_dimensions(xc_1_aggregate, (0, 2, 1))
        output = K.batch_dot(xc_2_aggregate, xc_1_aggregate)    # batch_size x wh x w'h'
        output = K.reshape(output, (-1, output_row, output_col, output_shape[-1]))
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'patch_size': self.kernel_size,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'stride': self.subsample,
                  'dim_ordering': self.dim_ordering}
#        base_config = super(Correlation_Layer, self).get_config()
        base_config = super(Normalized_Correlation_Layer, self).get_config();
        return dict(list(base_config.items()) + list(config.items()))

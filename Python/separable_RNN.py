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
#                     Separable Simple RNN Layer
# =============================================================================


class Separable_SimpleRNN(Recurrent):
    '''This is the vanilla RNN from the separable idea,
       this layer only includes the recurrence term,
       therefore, 
       the input to this layer should be directly from linear convolution,
       or should be from Batch Normalization.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        inner_init: initialization function of the inner cells.
        activation: activation function.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    '''

    def __init__(self, output_dim,
                 inner_init='orthogonal',
                 activation='tanh',
                 U_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(Separable_SimpleRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.U = self.add_weight((self.output_dim, self.output_dim),
                                 initializer=self.inner_init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, it needs to know '
                            'its batch size. Specify the batch size '
                            'of your input tensors: \n'
                            '- If using a Sequential model, '
                            'specify the batch size by passing '
                            'a `batch_input_shape` '
                            'argument to your first layer.\n'
                            '- If using the functional API, specify '
                            'the time dimension by passing a '
                            '`batch_shape` argument to your Input layer.')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]

        output = self.activation(x * B_W + K.dot(prev_output * B_U, self.U))
        return output, [output]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(Separable_SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

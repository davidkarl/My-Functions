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
#                     Layer Normalization RNN Layer
# =============================================================================


class LN_SimpleRNN(Recurrent):
    '''Fully-connected RNN where the output is to be fed back to input.
       This is the version with Layer Normalization.
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        separable: True or False (default), 
        if True, the input to this layer should be the result from linear convolution. 
    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
    '''

    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', beta_init='zero', gamma_init='one',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 gamma_regularizer=None, beta_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        self.epsilon = 1e-5
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(LN_SimpleRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer)
        self.U = self.add_weight((self.output_dim, self.output_dim),
                                 initializer=self.inner_init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer)
        self.b = self.add_weight((self.output_dim,),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer)

        self.gamma = self.add_weight((self.input_dim,),
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name))
        self.beta = self.add_weight((self.input_dim,),
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name))

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
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                          input_dim, self.output_dim,
                                          timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]

        if self.consume_less == 'gpu':
            h = K.dot(x * B_W, self.W) + self.b
        else:
            h = x
        out = h + K.dot(prev_output * B_U, self.U)
        mean = K.mean(out, axis=-1, keepdims=True)
        std = K.std(out, axis=-1, keepdims=True)
        output_normed = self.activation((out - mean) / (std + self.epsilon))
        output = self.gamma * output_normed + self.beta
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
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(LN_SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

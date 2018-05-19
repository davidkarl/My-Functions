# deep_speckles


######   IMPORT RELEVANT LIBRARIES: ###########
#(1). Main Modules
from __future__ import print_function
import keras
from keras import backend as K
import tensorflow as tf
#import cv2    NEED TO USE PIP INSTALL!!!!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight');
import os
import datetime
import time
import sys
import nltk
import pylab
import collections
import re
from csv import reader
import tarfile
from pandas import read_csv
from pandas import Series
import collections
#counter = collections.Counter()
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.ar_model import AR
import argparse
import pydot
import graphviz
import pydot_ng
length = len

from numpy import power as power
from numpy import exp as exp
from pylab import imshow, pause, draw, title, axes, ylabel, ylim, yticks, xlabel, xlim, xticks
from pylab import colorbar, colormaps, colors, subplot, suptitle, plot


######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
###########   Keras Functions: #########3
#(1). Models:
from keras.models import Sequential, Model, Input #,InputLayer
####################################################################################################################################
#(2). Layers
# Convolutions / Upsampling
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, ConvLSTM2D, ConvRecurrent2D
from keras.layers import Deconv2D, Deconv3D
from keras.layers import AtrousConvolution1D, AtrousConvolution2D
from keras.layers import UpSampling1D, UpSampling2D, UpSampling3D
# Misceleneous
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Embedding
from keras.layers import GaussianNoise, Bidirectional, Highway
# Padding
from keras.layers import ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
# Dropouts
from keras.layers import SpatialDropout1D, SpatialDropout2D, SpatialDropout3D, Dropout, AlphaDropout, GaussianDropout
# Pooling
from keras.layers import Average, AveragePooling1D, AveragePooling2D, AveragePooling3D
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.layers import Cropping1D, Cropping2D, Cropping3D
# Recurrent
from keras.layers import LSTM, GRU, SimpleRNN
# Relu's
from keras.layers import ELU, LeakyReLU, PReLU
# Custom:
from keras.layers import Lambda
# Shapes
from keras.layers import Merge, RepeatVector, Reshape, TimeDistributed, Bidirectional, Concatenate
from keras.layers import merge, concatenate
#(3). Optimizers
from keras.optimizers import Adam,SGD,RMSprop,Adagrad,Nadam,TFOptimizer
#(4). Callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, BaseLogger, EarlyStopping, LambdaCallback, CSVLogger
#(5). Generators
from keras.preprocessing.image import ImageDataGenerator
#(6). Regularizers
from keras.regularizers import l2
#(7). Normalization (maybe add weight normalization, layer normalization, SELU)
#from keras.layers.normalization import BatchNormalization
#(8). Preprocessing
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
#(9). Objectives
from keras.objectives import binary_crossentropy, categorical_crossentropy,cosine_proximity,categorical_hinge
from keras.objectives import hinge, logcosh, kullback_leibler_divergence, mean_absolute_error, mean_squared_error
from keras.objectives import mean_squared_logarithmic_error, sparse_categorical_crossentropy, squared_hinge
#(10). Utils
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.utils.data_utils import get_file, random, absolute_import
from keras.utils.conv_utils import conv_input_length, conv_output_length
from keras.utils.generic_utils import Progbar
from keras.utils.io_utils import HDF5Matrix
from keras.utils import plot_model
#(11). Models
from keras.models import copy, h5py, json, load_model, model_from_json, model_from_config, save_model
#(12). Activations:
from keras.activations import elu, relu, selu, linear, sigmoid, softmax, softplus, softsign, tanh
#(13). Backend:
#from keras.backend import concatenate
#(14). Metrics: mse, mae, mape, cosine
#Mean Squared Error: mean_squared_error, MSE or mse
#Mean Absolute Error: mean_absolute_error, MAE, mae
#Mean Absolute Percentage Error: mean_absolute_percentage_error, MAPE, mape
#Cosine Proximity: cosine_proximity, cosine

##Example of Usage:
#model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])
#history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)
#pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['mean_absolute_error'])
#pyplot.plot(history.history['mean_absolute_percentage_error'])
#pyplot.plot(history.history['cosine_proximity'])



######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
#########  Numpy Functions: ############
#(1). Arrays & Matrices
from numpy import array, arange, asarray, asmatrix, atleast_1d, atleast_2d, atleast_3d, copy
#(2). Apply operators:
from numpy import apply_along_axis, apply_over_axes, transpose
#(3). Finding indices/elements wanted
from numpy import amax, amin, argmin, argmax, argwhere, argsort, where, who
from numpy import equal,greater_equal, greater, not_equal, less_equal
from numpy import min, max
#(4). Mathematical Operations
from numpy import absolute, add, average, exp, exp2, log, log10, log2, mod, real, imag, sqrt, square
from numpy import floor, angle, conj, unwrap
from numpy import mean, median, average, cumsum, std, diff, clip
#(5). Linspace, Meshgrid:
from numpy import meshgrid, linspace, logspace, roll#, roll_axis
#(6). Shape Related:
from numpy import reshape, resize, shape, newaxis, rot90, flip, fliplr, flipud, expand_dims, left_shift
from numpy import squeeze, moveaxis #flatten
#(7). Stacking arrays
from numpy import hstack, vstack, hsplit, column_stack, row_stack, repeat
#(8). Initiaizing certain arrays
from numpy import empty, empty_like, dtype, eye, zeros, ones, zeros_like, ones_like
#(9). Splitting arrays
from numpy import select, split, unique, choose
#(10). Histograms
from numpy import histogram, histogram2d
#(11). FFT:
from numpy.fft import fft, fft2, fftfreq, fftn, fftshift
#(12). Linear-Algebra
from numpy.linalg import cholesky, eig, inv, lstsq, matrix_rank, norm, pinv, qr, solve, svd
#(13). Loading and Saving
from numpy import load, save, fromfile
#(14). Random
from numpy.random import choice, permutation, shuffle
from numpy.random import multivariate_normal, normal, rand, randint, randn, uniform, rayleigh
#(15). Import all
from numpy.fft import *
from numpy.linalg import *
from numpy import *
from numpy.random import *
from numpy import power as power
from numpy import exp as exp
#(16). Define "end" to make it more like matlab:
end = -1;
start = -1; # Matlab: vec(1:end)
            # Python: vec(start+1:end)

######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################


######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######### Math/Random Functions/Constants: #########
from math import sin
from math import pi
#from math import exp
from math import pow
from random import random, randint, uniform, randrange, choice, gauss, shuffle
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################


######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######### SKLEARN: #########
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from scipy import signal
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

########### pyqtgraph stuff #############
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE
from pyqtgraph.ptime import time
from pyqtgraph import mkPen
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################



######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
############ PERSONAL IMPORTS ###########

#from importlib import reload
#import things_to_import
#things_to_import = reload(things_to_import)
#from things_to_import import *

from importlib import reload
import int_range, int_arange, mat_range, matlab_arange, my_linspace, my_linspace_int,importlib
from int_range import *
from int_arange import *
from mat_range import *
from matlab_arange import *
from my_linspace import *
from my_linspace_int import *
import get_center_number_of_pixels
get_center_number_of_pixels = reload(get_center_number_of_pixels);
from get_center_number_of_pixels import *
import get_speckle_sequences
get_speckle_sequences = reload(get_speckle_sequences);
from get_speckle_sequences import *
import tic_toc
from tic_toc import *
import search_file
from search_file import *
import show_matrices_video
from show_matrices_video import *
import modify_speckle_sequences
from modify_speckle_sequences import *
import get_speckle_sequences_full
from get_speckle_sequences_full import *
import klepto_functions
from klepto_functions import *
import ctypes  # An included library with Python install.
def message_box(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
import pymsgbox
from pymsgbox import alert as alert_box
from pymsgbox import confirm as confirm_box
from pymsgbox import prompt as prompt_box
# alert_box('alert text','alert title')
# response = pymsgbox.prompt('What is your name?')

####################################################################################################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################


########### Re-Initialize weights in model ##############
def reinitialize_model_weights(keras_model):
    session = K.get_session()
    for layer in keras_model.layers:
     for v in layer.__dict__:
         v_arg = getattr(layer,v)
         if hasattr(v_arg,'initializer'):
             initializer_method = getattr(v_arg, 'initializer')
             initializer_method.run(session=session)



#####################################################################################################################################################################
#####################################################################################################################################################################
########### BASIC CNN LAYERS FOR A SINGLE IMAGE FEATURE EXTRACTION ##############
def vision_block_CONV2D(input_layer, number_of_filters_vec, kernel_size_vec, \
                 flag_dilated_convolution_vec, \
                 flag_resnet_vec, flag_size_1_convolution_on_shortcut_in_resnet_vec, \
                 flag_batch_normalization_vec, \
                 flag_size_1_convolution_after_2D_convolution_vec, flag_batch_normalization_after_size_1_convolution_vec, \
                 activation_type_str_vec):

    number_of_kernel_sizes_in_layer = np.size(kernel_size_vec);

    for kernel_size_counter in arange(0,number_of_kernel_sizes_in_layer,1):
        #Get parameters for each kernel size filters in layer:
        number_of_filters = number_of_filters_vec[kernel_size_counter];
        kernel_size = kernel_size_vec[kernel_size_counter];
        flag_dilated_convolution = flag_dilated_convolution_vec[kernel_size_counter];
        flag_resnet = flag_resnet_vec[kernel_size_counter];
        flag_size_1_convolution_on_shortcut_in_resnet = flag_size_1_convolution_on_shortcut_in_resnet_vec[kernel_size_counter];
        flag_batch_normalization = flag_batch_normalization_vec[kernel_size_counter];
        flag_size_1_convolution_after_2D_convolution = flag_size_1_convolution_after_2D_convolution_vec[kernel_size_counter];
        flag_batch_normalization_after_size_1_convolution = flag_batch_normalization_after_size_1_convolution_vec[kernel_size_counter];
        activation_type_str = activation_type_str_vec[kernel_size_counter];

        kernel_size = (kernel_size,kernel_size);


        if flag_dilated_convolution==1:
            vision_block_current_kernel_size = AtrousConvolution2D(number_of_filters, kernel_size, atrous_rate=dilation_rate,border_mode='same')(input_layer);
        else:
            vision_block_current_kernel_size = Conv2D(number_of_filters, kernel_size, padding='same')(input_layer);

        if flag_batch_normalization==1:
            vision_block_current_kernel_size = BatchNormalization()(vision_block_current_kernel_size);

        vision_block_current_kernel_size = Activation(activation_type_str)(vision_block_current_kernel_size);

        if flag_size_1_convolution_after_2D_convolution==1:
            vision_block_current_kernel_size = Conv2D(number_of_filters, 1, padding='same')(vision_block_current_kernel_size);
            if flag_batch_normalization_after_size_1_convolution==1:
                vision_block_current_kernel_size = BatchNormalization()(vision_block_current_kernel_size);
            vision_block_current_kernel_size = Activation(activation_type_str)(vision_block_current_kernel_size);


        if flag_resnet==1:
            if flag_size_1_convolution_on_shortcut_in_resnet==1:
                input_layer = Conv2D(number_of_filters,1,border_mode='same');
                if flag_batch_normalization_after_size_1_convolution==1:
                    input_layer = BatchNormalization()(input_layer);
            vision_block_current_kernel_size = merge([vision_block_current_kernel_size,input_layer],mode='sum')


        if kernel_size_counter == 0:
            vision_block = vision_block_current_kernel_size;
        else:
            vision_block = keras.layers.concatenate([vision_block,vision_block_current_kernel_size]);

    #END OF KERNEL SIZE FOR LOOP

    return vision_block
#####################################################################################################################################################################
#####################################################################################################################################################################


def build_convolutional_network(input_layer, parameter_dictionary_input):
    #Here i need to decide what flexibility i want. i will loop over the keys and look for keys that begin with what i'm looking for.
    #this will also allow me to use default values.
    #I could have used **kw however my input dictionary keys have an EXTRA _string at the end....

    #Input variables names needed in the function:
    input_variables_names_list = ['number_of_convolutional_model_layers',
                                  'number_of_filters_list',
                                  'kernel_size_list',
                                  'flag_dilated_convolution_list',
                                  'flag_resnet_list',
                                  'flag_size_1_convolution_on_shortcut_in_resnet_list',
                                  'flag_batch_normalization_list',
                                  'flag_size_1_convolution_after_2D_convolution_list',
                                  'flag_batch_normalization_after_size_1_convolution_list',
                                  'activation_type_str_list']

    #Build function's needed parameters dictionary:
    parameter_dictionary = dict();
    for k,v in parameter_dictionary_input.items():
        for i in input_variables_names_list:
            if k.startswith(i):
                parameter_dictionary[i] = v;

    #Build Convolutional Neural Network:
    convolutional_network = input_layer;
    for layer_counter in arange(0,parameter_dictionary['number_of_convolutional_model_layers'],1):
        current_number_of_filters_in_layer = parameter_dictionary['number_of_filters_list'][layer_counter];
        current_kernel_size_in_layer = parameter_dictionary['kernel_size_list'][layer_counter];
        current_flag_dilated_convolution_in_layer = parameter_dictionary['flag_dilated_convolution_list'][layer_counter];
        current_flag_resnet_in_layer = parameter_dictionary['flag_resnet_list'][layer_counter];
        current_flag_size_1_convolution_on_shortcut_in_resnet_in_layer = parameter_dictionary['flag_size_1_convolution_on_shortcut_in_resnet_list'][layer_counter];
        current_flag_batch_normalization_in_layer = parameter_dictionary['flag_batch_normalization_list'][layer_counter];
        current_flag_size_1_convolution_after_2D_convolution_in_layer = parameter_dictionary['flag_size_1_convolution_after_2D_convolution_list'][layer_counter];
        current_flag_batch_normalization_after_size_1_convolution_in_layer = parameter_dictionary['flag_batch_normalization_after_size_1_convolution_list'][layer_counter];
        current_activation_type_str_in_layer = parameter_dictionary['activation_type_str_list'][layer_counter];

        current_number_of_kernel_sizes_in_layer = np.size(kernel_size_list[layer_counter]);

        convolutional_network = vision_block_CONV2D(convolutional_network,
                                                 current_number_of_filters_in_layer,
                                                 current_kernel_size_in_layer,
                                                 current_flag_dilated_convolution_in_layer,
                                                 current_flag_resnet_in_layer,
                                                 current_flag_size_1_convolution_on_shortcut_in_resnet_in_layer,
                                                 current_flag_batch_normalization_in_layer,
                                                 current_flag_size_1_convolution_after_2D_convolution_in_layer,
                                                 current_flag_batch_normalization_after_size_1_convolution_in_layer,
                                                 current_activation_type_str_in_layer,
                                                 );
    #END of layers loop
    return convolutional_network;



#####################################################################################################################################################################
#####################################################################################################################################################################
##### CUSTOM LAYER: #####
#There are only three methods you need to implement:

#(1). build(input_shape): this is where you will define your weights.
#                    This method must set self.built = True, which can be done by
#                    calling super([Layer], self).build().
#(2). call(x): this is where the layer's logic lives. Unless you want your layer to support masking,
#         you only have to care about the first argument passed to call: the input tensor.
#(3). compute_output_shape(input_shape): in case your layer modifies the shape of its input, you should
#                                   specify here the shape transformation logic.
#                                   This allows Keras to do automatic shape inference.

##Usage Example:
#from keras import backend as K
#from keras.engine.topology import Layer
#import numpy as np
#class MyLayer(Layer):
#
#    def __init__(self, output_dim, **kwargs):
#        self.output_dim = output_dim
#        super(MyLayer, self).__init__(**kwargs)
#
#    def build(self, input_shape):
#        # Create a trainable weight variable for this layer.
#        self.kernel = self.add_weight(name='kernel',
#                                      shape=(input_shape[1], self.output_dim),
#                                      initializer='uniform',
#                                      trainable=True)
#        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!
#
#    def call(self, x):
#        return K.dot(x, self.kernel)
#
#    def compute_output_shape(self, input_shape):
#        return (input_shape[0], self.output_dim)
#

#####################################################################################################################################################################
#####################################################################################################################################################################





#####################################################################################################################################################################
#####################################################################################################################################################################
########### BASIC MLP BLOCK ##############
def mlp_block(input_layer, number_of_neurons_per_layer, activation_functions_per_layer):
    number_of_layers = len(np.array(number_of_neurons_per_layer));
    input_layer = Flatten()(input_layer);

    for layer_counter in arange(0,number_of_layers,1):
        input_layer = Dense(units = number_of_neurons_per_layer[layer_counter],
                            activation = activation_functions_per_layer[layer_counter])(input_layer);

    return input_layer
#####################################################################################################################################################################
#####################################################################################################################################################################



################################################################################################################################################################################################################
################################################################################################################################################################################################################
###############################################################################################################################################################################################################
########### Data Generator: ##############

#The data generator should output several images for the time distributed model:
class DataGenerator(object):

  #'Generates data for Keras'
  def __init__(self,
               ROI=32,
               speckle_size = 5,
               epp = 10000,
               readout_noise = 140,
               background = 0,
               max_shift = 0.1,
               number_of_time_steps = 4,
               batch_size = 32, \
               flag_single_number_or_optical_flow_prediction = 1, \
               constant_decorrelation_fraction = 0,
               Fs = 5500,\
               flag_center_each_speckles_independently_or_as_batch_example = 2, \
               flag_stretch_each_speckles_independently_or_as_batch_example = 2, \
               std_wanted = 1, \
               flag_stateful_batch = 1, \
               flag_data_generation_online_or_from_file = 1,\
               file_name_X = 'bla', \
               simulation_parameters_klepto_file_name = 'blabla', \
               flag_shuffle = False, \
               flag_get_data_as_function_or_as_generator = 1):

      #Act as a function to call or as a python generator:
      self.flag_get_data_as_function_or_as_generator = flag_get_data_as_function_or_as_generator;
      self.flag_data_generation_online_or_from_file = flag_data_generation_online_or_from_file; #can i even add such a variable????
      self.flag_EOF = 0;
      self.flag_use_shot_noise = 1; #INSERT INTO FUNCTION INPUT ARGUMENTS
      self.cross_talk = 0.35;
      self.stateful_batch_size = 32;


      #Get Class Varaibles depending on whether i generate samples online or from file:
      if flag_data_generation_online_or_from_file == 1:
          #Online (given input parameters are relevant):

          #(1). Macro Parameters:
          # 'Initialization'
          self.ROI = int(ROI);
          self.number_of_time_steps = int(number_of_time_steps);
          self.batch_size = int(batch_size);
          self.flag_shuffle = flag_shuffle; #i don't see any reason for this to be true as i do this mehanically
          self.flag_single_number_or_optical_flow_prediction = flag_single_number_or_optical_flow_prediction;

          #(2). Speckles Sequence Simulation parameters:
          self.constant_decorrelation_fraction = constant_decorrelation_fraction; #this is mostly important for turbulence
          self.Fs = Fs

          #(3). Specific Speckles parameters:
          self.speckle_size = speckle_size;
          self.epp = epp
          self.readout_noise = readout_noise;
          self.background = background;
          self.max_shift = max_shift;
          self.flag_center_each_speckles_independently_or_as_batch_example = flag_center_each_speckles_independently_or_as_batch_example
          self.flag_stretch_each_speckles_independently_or_as_batch_example = flag_stretch_each_speckles_independently_or_as_batch_example
          self.std_wanted = std_wanted
          self.flag_stateful_batch = flag_stateful_batch;

          #(4). From file parameters (just for completeness):
          self.flag_center_each_speckles_independently_or_as_batch_example_from_file = None;
          self.flag_stretch_each_speckles_independently_or_as_batch_example_from_file = None;
          self.std_wanted_from_file = None;

          #(5).Initialize {X,y}:
          self.X = None;
          self.y = None;

      if flag_data_generation_online_or_from_file == 2:
          #from file:

          #(1). Choose online data generation or from file
          #MAKE THIS MORE ROBUST:
          self.file_name_X = file_name_X;
          self.file_name_y = 'speckle_shifts' + file_name_X[16:];
          self.file_name_simulation_parameters = simulation_parameters_klepto_file_name;
          #open binary files to be read:
          self.fid_X = open(self.file_name_X,'rb');
          self.fid_y = open(self.file_name_y,'rb');
          db = file_archive(self.file_name_simulation_parameters + '.txt');
          db = file_archive(simulation_parameters_klepto_file_name + '.txt');
          db.load();

          # db_keys_list = list(db.keys());
          # db_values_list = list(db.values());
          # for db_key,db_value in zip(db_keys_list,db_values_list):
          #     print(db_key)


          #Assign the object (self.parameter) the relevant parameters in the db klepto file:
          db_keys_list = list(db.keys());
          db_values_list = list(db.values());
          execution_string = '';
          for db_key,db_value in zip(db_keys_list,db_values_list):
              if type(v) != str:
                  execution_string = execution_string + '\n' + 'self.' + db_key + '=' + str(db_value);
              else:
                  execution_string = execution_string + '\n' + 'self.' + db_key + '=\'' + db_value + '\'';
          exec(execution_string);



          #(2). get variables from file name:
          # self.parameters_encoded_in_file_name_string = file_name_X[16:];
          # self.parameters_list = self.parameters_encoded_in_file_name_string.split('$');
          # self.parameters_list.pop(0);
          self.ROI = int(db['ROI']);
          self.speckle_size = float(db['speckle_size']);
          self.epp = float(db['epp']);
          self.readout_noise = float(db['readout_noise']);
          self.background = float(db['background']);
          self.number_of_time_steps = int(db['number_of_time_steps']);
          self.flag_single_number_or_optical_flow_prediction = int(db['flag_single_number_or_optical_flow_prediction']);
          self.constant_decorrelation_fraction = float(db['constant_decorrelation_fraction']);
          self.Fs = float(db['Fs']);
          self.batch_size = int(db['batch_size_file']);
          self.data_type = db['data_type'];

          self.flag_center_each_speckles_independently_or_as_batch_example_from_file = int(db['flag_center_each_speckles_independently_or_as_batch_example_from_file']);
          self.flag_stretch_each_speckles_independently_or_as_batch_example_from_file = int(db['flag_stretch_each_speckles_independently_or_as_batch_example_from_file']);
          self.std_wanted_from_file = int(db['std_wanted_from_file']);
          self.flag_stateful_batch = int(db['flag_stateful_batch']);

          #(3)Post Processing steps i get as function/class inputs:
          self.flag_center_each_speckles_independently_or_as_batch_example = flag_center_each_speckles_independently_or_as_batch_example
          self.flag_stretch_each_speckles_independently_or_as_batch_example = flag_stretch_each_speckles_independently_or_as_batch_example
          self.std_wanted = std_wanted

          #(4).Initialize {X,y}:
          self.X = None;
          self.y = None;

          #(5).Other stuff:
          self.max_shift = max_shift;


          #END OF CLASS init() function WITH VARIABLE ASSIGNMENT.


  def generate_function(self,number_of_batches = 32):
      'Generates batches of samples'
      # Infinite loop
      self.flag_EOF = 0;
      while 1==1: #Notice the change from while to if to keep it consistent with the iterator

          #Initialize Variables:
          #(1). Image Sequence:
          if (self.X is None and self.y is None):
              flag_reinitialize_xy = 1;
          elif (shape(self.y)[0] != number_of_batches):
              flag_reinitialize_xy = 1;

          if flag_reinitialize_xy ==1:
              #now further check that there hasn't been a change in desired number of batches to avoid
              #unnecessary initializations
              self.X = np.empty((int(self.batch_size*number_of_batches), self.number_of_time_steps, self.ROI, self.ROI, 1)); #Images
              #(2). Labels:
              if self.flag_single_number_or_optical_flow_prediction == 1:
                  self.y = np.empty((int(self.batch_size*number_of_batches),2)); #Shift {x&y} between frame N/2 and N/2+1
              else:
                  self.y = np.empty((int(self.batch_size*number_of_batches),self.ROI,self.ROI,2)); #Optical flow (2 image sized predictions, one for x and one for y)



          #(2). Decide Whether online or from file and generate batch:
          if self.flag_data_generation_online_or_from_file==1:
             ####### Online

             #Online - this means i got the meaningful variables as inputs into the generate_function
             current_sample = 0;
             for batch_counter in arange(0,number_of_batches,1):
                 [X_current,y_current] = get_speckle_sequences_full(self.ROI,
                                                                    self.speckle_size,
                                                                    self.max_shift,
                                                                    self.number_of_time_steps,
                                                                    self.batch_size, \
                                                                    self.flag_single_number_or_optical_flow_prediction, \
                                                                    self.constant_decorrelation_fraction,
                                                                    self.Fs,\
                                                                    self.flag_stateful_batch,\
                                                                    data_type = 'f');


                 #Modify (center, stretch, add noise):
                 X_current = modify_speckle_sequences(X_current,\
                                                      self.epp,\
                                                      self.readout_noise,\
                                                      self.background,\
                                                      self.flag_use_shot_noise,\
                                                      self.cross_talk,\
                                                      self.flag_center_each_speckles_independently_or_as_batch_example,\
                                                      self.flag_stretch_each_speckles_independently_or_as_batch_example,\
                                                      self.std_wanted,\
                                                      self.flag_stateful_batch,\
                                                      self.stateful_batch_size);

                 self.X[current_sample:current_sample+self.batch_size,:,:,:,:] = X_current;
                 self.y[current_sample:current_sample+self.batch_size,:] = y_current;
                 current_sample = current_sample + self.batch_size;




          ###### From File:
          elif self.flag_data_generation_online_or_from_file==2:
             #From File:

             #Get number of samples and elements and elements shape to read from file:
             number_of_batch_samples = int(number_of_batches * self.batch_size);
             number_of_images_to_read = int(number_of_batch_samples * self.number_of_time_steps);
             ROI_shape = (int(self.ROI),int(self.ROI));
             mat_shape = np.append(number_of_batch_samples,int(number_of_time_steps));
             mat_shape = np.append(mat_shape,ROI_shape)
             mat_shape = np.append(mat_shape,1)
             single_image_number_of_elements = np.prod(ROI_shape);
             total_images_number_of_elements = single_image_number_of_elements*number_of_images_to_read;

             # #Go to place last stopped at if wanted:
             # if flag_seek_in_binary_file == 1:
             #     self.fid_X.seek(number_of_fid_X_elements_so_far);
             #     self.fid_y.seek(number_of_fid_y_elements_so_far);

             #Read speckle images from file:
             self.X = np.fromfile(self.fid_X,'f',count=total_images_number_of_elements);
             X_length = length(self.X);
             self.y = np.fromfile(self.fid_y,'f',count=number_of_batch_samples*2)
             y_length = length(self.y);

             #Check we are not in EOF and if so rewind and add samples:
             if size(self.y) < number_of_batch_samples * 2:
                 #Return EOF as hachana lemazgan:
                 self.flag_EOF = 1;
                 #Rewind FID and add the number of examples need to complete number of batches expected:
                 self.fid_X.seek(0);
                 self.fid_y.seek(0);
                 X_complete = np.fromfile(self.fid_X,'f',count=total_images_number_of_elements-X_length);
                 y_complete = np.fromfile(self.fid_y,'f',count=number_of_batch_samples*2-y_length);
                 self.X = np.hstack((self.X,X_complete));
                 self.y = np.hstack((self.y,y_complete));

             #Reshape X and y:
             self.X = self.X.reshape(mat_shape);
             self.y = self.y.reshape((number_of_batch_samples,2));

             #Post-Process if called for:
             if self.flag_center_each_speckles_independently_or_as_batch_example_from_file == 3 and self.flag_stretch_each_speckles_independently_or_as_batch_example_from_file == 3:
                 #Modify (center, stretch, add noise):
                 self.X = modify_speckle_sequences(self.X,\
                                                   self.epp,\
                                                   self.readout_noise,\
                                                   self.background,\
                                                   self.flag_use_shot_noise,\
                                                   self.cross_talk,\
                                                   self.flag_center_each_speckles_independently_or_as_batch_example,\
                                                   self.flag_stretch_each_speckles_independently_or_as_batch_example,\
                                                   self.std_wanted,\
                                                   self.flag_stateful_batch,\
                                                   self.stateful_batch_size);


          #NOW that i got {X,y} i need to decide how to get them - whether to return them like a function
          #call or with a "yield" keyword like a python generator:
          if self.flag_get_data_as_function_or_as_generator == 1:
              #As function:
              return self.X,self.y,self.flag_EOF;



          #END of if flag_data_generation_online_or_from_file==1
      #END of while 1:
  #END of generator function in DataGenerator class

  #i should probably write another function idential to generate_function but as generator. this version is
  #proably slow.
  def generate_function_as_python_generator(self):
     yield generate_function(self,number_of_batches = self.batch_size);

  #Taking Care of fids (file i.d):
  def close_fids(self):
      self.fid_X.close()
      self.fid_y.close()

  def open_fids(self):
      self.fid_X = open(self.file_name_X,'rb');
      self.fid_y = open(self.file_name_y,'rb');

  def reset_fids(self):
      self.fid_X.close();
      self.fid_y.close();
      self.fid_X = open(self.file_name_X,'rb');
      self.fid_y = open(self.file_name_y,'rb');

#END OF DataGenerator class


################################################################################################################################################################################################################
################################################################################################################################################################################################################
###############################################################################################################################################################################################################

############ CALLBACKS: #############

#ModelCheckpoint(filepath,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto',period=1);
#####   filepath: weights.{epoch:02d}_{val_loss:.2f}.hdf5
#EarlyStopping(monitor='val_loss',min_delta=0,times_to_wait_before_learning_rate_update=0,vrbose=0,mode='auto');
#ReduceOnPlateau(monitor='val_loss',factor=0.1,times_to_wait_before_learning_rate_update=10,verbose=0,mode='auto',epsilon=0.001,cool_down=0,min_lr=0);
#LambdaCallback(on_epoch_begin=None,on_epoch_end=None,on_batch_begin=None,on_batch_end=None,on_train_begin=None,on_train_end=None)
#TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
#            write_grads=False, write_images=False,
#            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#### if you have installed tensorflow with pp you should be able to launce tensorboard from the command line:
#### tensorboard --logdir=/full_path_to_your_logs

class custom_callback(keras.callbacks.Callback):
    # logs include: loss, acc, val_loss, val_acc

    def __init__(self, times_to_wait_before_learning_rate_update=0, learning_rate_reduction_factor=0.1, number_of_learning_rate_reductions_before_learning_stops=10, verbose=1):
        super(keras.callbacks.Callback, self).__init__()

        #Keep track of learning rate:
        self.times_to_wait_before_learning_rate_update = times_to_wait_before_learning_rate_update
        self.counter_to_learning_rate_update = 0
        self.best_validation_loss_so_far = -1.
        self.learning_rate_reduction_factor = learning_rate_reduction_factor
        self.counter_number_of_times_learning_rate_was_reducted = 0
        self.number_of_learning_rate_reductions_before_learning_stops = number_of_learning_rate_reductions_before_learning_stops
        self.verbose = verbose

        #Keep track of errors over time:
        self.error_over_time_validation = [];
        self.error_over_time_training = [];
        self.number_of_batches_before_validation = number_of_batches_before_validation; #not used as of now
        self.batch_counter = 0;

    #Train:
    def on_train_begin(self,logs={}):
        1;

    def on_train_end(self,logs={}):
        1;

    #Batch:
    def on_batch_begin(self,batch,logs={}):
        1;
    def on_batch_end(self,batch,logs={}):
        #Append training error over time each time a batch is finished processing:
        #(if i'm using gpus, do i have to make anything different?)
        self.error_over_time_training.append(logs.get('loss'));


    #Epoch:
    def on_epoch_begin(self,epoch,logs={}):
        1;
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('val_loss') is not None):
            #Parameters:
            1;




################################################################################################################################################################################################################
################################################################################################################################################################################################################
################################################################################################################################################################################################################
####### AUXILIARY TensorFlow/Keras FUNCTIONS: ########

def clip_shift_layer(predicted_shifts, max_shift=1):
#    predicted_shifts[(predicted_shifted > max_shift)] = max_shift;
    return K.clip(predicted_shifts,-max_shift,max_shift);

def custom_loss_function(predicted_shifts, true_shifts):
    #if i predict images
    max_shift = max_shift_global;
    predicted_x = predicted_shifts[0];
    predicted_y = predicted_shifts[1];
    true_x = true_shifts[0];
    true_y = true_shifts[1];
    if flag_clip_loss_values == 1:
        difference_clipped = K.clip(K.abs(predicted_shifts-true_shifts),min_value=-max_shift,max_value=max_shift);
    else:
        difference_clipped = K.abs(predicted_shifts-true_shifts);

    return K.mean(K.square(predicted_shifts - true_shifts), axis=-1)


def change_learning_rate(keras_model,factor):
    K.set_value(keras_model.optimizer.lr,
            K.get_value(keras_model.optimizer.lr) * factor);


#def TB(cleanup=False): #need to take care of this or more properly - integrated tensorflow explicitly in script
#    import webbrowser
#    webbrowser.open('http://127.0.1.1:6006')
#
#    !tensorboard --logdir="logs"
#
#    if cleanup:
#        !rm -R logs/



################################################################################################################################################################################################################
################################################################################################################################################################################################################
##############################################
#Plot Graph - DOESN'T WORK AS OF NOW!!!!. should use keras' plot_graph and tensorboard instead.
from IPython.display import display, HTML
from PIL import Image
import requests
from io import BytesIO

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def


#DOESN'T WORK FOR NOW:
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


################################################################################################################################################################################################################
################################################################################################################################################################################################################
#########################################################################################################################################################################################################


#######    POSSIBLE PREPARED MODELS   ########
# Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)






################################################################################################################################################################################################################
########################## BUILD THE NETWORKS: ####################
########################## network parameters: ####################
K.clear_session() #this is in order to make layer names consistent each time i create a model so that


#Get baseline variables before starting the script:
exec(save_tic(1));

###########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################

#####    MODEL NAMES #####
#(1). Should i start new training or continue from previous keras saved model:
flag_start_new_training_or_continue_training_previous_model = 1; #1=start new training, 2=continue from previous model
#(2). If i'm using previous version of model - what to do about
flag_previous_model_training_options = 1; #1=take only architecture and retrain,
                                          #2=take total model and continue previous script training to completion
                                          #3=take total model and add current script batch definitions (TRAINING CHNAGE)
if flag_start_new_training_or_continue_training_previous_model == 1: flag_previous_model_training_options=0;
#(3). Basic Model name:
basic_model_name = 'DeepSpeckles_TimeDistributed';
#(4). Training Directory (where to look for and put the training data .bin file):
training_directory = 'C:/Users\master\.spyder-py3\KERAS'


if flag_start_new_training_or_continue_training_previous_model == 2:
    #Use Previous Model in some sort of way (only architecture, train script to completion, add current script batches (training change) ):

    #(1). Keras Model (architecture + weights) name to continue training from:
    model_name_to_continue_training_from = 'DeepSpeckles_TimeDistributed_Version_29_TrainingChanges_0_ValidationEpoch_2_WeightInitialization_0';
    #(2). Extract Relevant Parameters From File Name:
    model_name_to_continue_training_from_split = model_name_to_continue_training_from.split('_');
    #(3). Assign the relevant parameters from the filename:
    model_version = model_name_to_continue_training_from_split[3];
    training_change = model_name_to_continue_training_from_split[5];
    validation_epoch = model_name_to_continue_training_from_split[7];
    weight_initiailization = model_name_to_continue_training_from_split[9];
    #(4). Get Whole Script Klepto file name:
    whole_script_klepto_file_name = '';
    for k in arange(0,6):
        whole_script_klepto_file_name = whole_script_klepto_file_name + model_name_to_continue_training_from_split[k] + '_';
    whole_script_klepto_file_name = whole_script_klepto_file_name + 'whole_script_klepto';
#End of if condition


#####   DATA FILE NAMES #######
#(4). Training Data .bin file and model name:
file_name_to_get_training_data_from_base = 'speckle_matrices_7';
file_name_to_get_training_data_from = file_name_to_get_training_data_from_base + '.bin'
simulation_parameters_klepto_file_name = file_name_to_get_training_data_from_base + '_klepto';




#####  BASIC PARAMETERS:  #######
########################################
flag_data_generation_online_or_from_file = 2          # Online data generation or from file.
########################################              # Maybe i should use keras K.randn or something and have all generation in graph
#Mixed Training: TO BE IMPLEMENTED!
params_mix_speckle_size = 0;
params_mix_epp = 0;
params_mix_readout_noise = 0;
params_mix_background = 0;
params_mix_constant_decorrelation_fraction = 0;
params_mix_ROI = 0;

#Speckle Parameters:
ROI = 32;
speckle_size = 5
number_of_time_steps = 4;
max_shift = 0.1
constant_decorrelation_fraction = 0

#Additive Noise Parameters:
epp = 10000 ;
readout_noise = 140;
background = 0;
cross_talk = 0.35; #To All neighboring pixels for now

#Temporal Parameters:
flag_white_noise_shifts = 1; #each sample with different random shift or have some temporal spectrum
Fs = 5500
noise_spectrum_power = -1; # (-1) -> 1/f, (-2) -> 1/f^2

#Cross Correlation Parameters:
flag_use_discrete_CC_as_input = 1;
flag_use_sub_pixel_CC_as_input = 1;
flag_use_deformation_map_as_input = 1;
#(1). Whole/Int Shifts Cross-Corrleation:
CC_size = 3;
#(2). Sub-Pixel Shifts using Fourier Phase Shifting:
sub_pixel_CC_size = 3;
CC_sub_pixel_size = 0.005;
flag_sub_pixel_shifts_map_type = 1; #(1)=difference map, (2)=cross correlation map
#(3). Deformation map:
deformation_map_size = ROI; #understand exactly how many pixels are valid.

#Sample Label Parameters (if samples are from file than they are set by that file not by the below parameters)
flag_single_number_or_optical_flow_prediction = 1;

#Stateful nature of the network/samples:
flag_stateful_batch = 0;
batch_size_stateful_if_online = 32; #if online and stateful this will be the batch_size_training TACHLES

#Optimizer:
#from keras.optimizers import Adam,SGD,RMSprop,Adagrad,Nadam,TFOptimizer
#Parameters:
flag_set_initial_learning_rate = 0;
initial_learning_rate = 0.1;
#Get optimizer:
optimizer = Adam();
if flag_set_initial_learning_rate == 1:
    K.set_value(optimizer.lr, initial_learning_rate);


#Learning rate Schedualer stuff:
#(1). Should i use learning rate reductions at all:
flag_use_learning_rate_schedualer = 1;
#(2). Number of times to wait WITHOUT SIGNIFICANT LOSS IMPROVEMENTS before reducing learning rate:
times_to_wait_before_learning_rate_update = 3;
#(3). Factor by which to reduce the learning rate:
learning_rate_reduction_factor = 0.1;
#(4). Numer of times to reduce learning rate before stopping to learn:
number_of_learning_rate_reductions_before_learning_stops = 10;
#(5). Loss improvement Tolerance (how significant must the improvement be before declaring a new best):
tolerance_around_best_score_linear = 0.05;
#(6). Counters:
counter_to_learning_rate_update = 0;
best_validation_loss_so_far = -1;
counter_number_of_times_learning_rate_was_reducted = 0;
batch_counter = 0;
verbose = 1;


#Training Parameters:
#(*). Use Custom Loss Function Or Not:
flag_use_custom_loss_function = 0;
flag_clip_loss_values = 0;
max_shift_global = 3; #max shift clipped at final loss function
#(*). Training Batches Output Format:
flag_center_each_speckles_independently_or_as_batch_example = 2
flag_stretch_each_speckles_independently_or_as_batch_example = 2
std_wanted = 1
flag_shuffle = False;


#If we use binary file to read data then i must OVERRIDE parameters defined above and use what is in the binary file:
#(*). NOTICE: i don't insist on batch_size_training = batch_size_file unless the data in file is stateful and then it matters!
if flag_data_generation_online_or_from_file == 2:
    #if from file:
    exec(load_variables_from_klepto_file(simulation_parameters_klepto_file_name));
#END if from_file


#Batch size Training and number of epochs:
exec(save_tic(5));
#(1). should be a whole multiply of the binary file defined batch size if stateful (prefered multiply will be 1 obviously)
if flag_stateful_batch == 0:
    #Every sample is independent!:
    batch_size_training = int(32*1);
else:
    #Batch samples are related!:
    if flag_data_generation_online_or_from_file == 1:
        #Online:
        batch_size_training = int(batch_size_stateful_if_online*1);
    else:
        #From File:
        batch_size_training = int(batch_size_file*1);
#(2). Number of different weight initializations:
number_of_different_weights_initializations = 5;
#(3). as much as possible in system RAM
number_of_batches_to_load_to_RAM = int(batch_size_training * 10);
#(4). number of times to load/generate data to RAM (Controlls Total amount of samples/data):
number_of_data_generations_or_loads_to_memory = int(10);
#(5). number of times to go through one generated data loaded to RAM (usually 1 unless i want to overfit):
epochs_per_generated_training_data = int(1);
#(6). num of batches to process before validation in fitting process:
number_of_batches_before_validation = int(10);
#(7). number of batches to pass to fit function (sets speeds a bit and number of examples before verbosity)
number_of_batches_to_give_to_fit_function = int(10);


# Validation:
#(1). Batch Size Validation:
if flag_stateful_batch == 1:
    #Stateful:
    if flag_data_generation_online_or_from_file == 1:
        #Online:
        batch_size_validation = int(batch_size_stateful_if_online*1);
    else:
        #From File:
        batch_size_validation = int(batch_size_file*1);
else:
    #Not stateful - set batch size however you want:
    batch_size_validation = int(32);
#(2). number of batches to check validation with:
number_of_validation_batches_to_check_with = 5;


# Method of data generation & FITTING!:
flag_get_data_as_function_or_as_generator = 1;
flag_method_of_fitting = 1; #(1). regular, very controlled .fit() , (2). Generator (Iterator)

#Make Sure that if flag_stateful==1 (which means that batch sizes must be whole multiples of binary file defined batch_size)!:
if flag_stateful_batch == 1 and flag_data_generation_online_or_from_file==2:
    #if data is from file AND stateful
    if mod(batch_size_validation,batch_size_file)!=0 or mod(batch_size_training,batch_size_file)!=0:
        print('batch_size_validation and batch_size_training must be a multiple of batch size when in flag_stateful_batch=1 mode');

#Save variables regarding training method and batch sizes
exec(save_toc(5,5));
############################################################################################################################################################################################################################################################################################




#########################################  NEURAL NETWORK ITSELF INSIDE OUT #######################################################################
##################################################################################################################################
#Save variables to future model comparison:
exec(save_tic(2));
############################################################################################################################################

#Network flags and auxiliary:
number_of_color_channels = 1;
flag_plot_model = 0;
flag_print_model_summary = 0;


#Dense Model on top of SINGLE IMAGE convolutional network:
flag_put_dense_on_top_of_convolutional_model = 1;
number_of_neurons_per_layer_convolutional_model = [10,10];
activation_function_per_layer_convolutional_model = ['linear','linear'];


#Dense Model on top of TIME DISTRIBUTED model:
flag_put_dense_on_top_of_time_distributed_image_model = 1; #This isn't really used(!) just for cleanness sake
                                                           #its functionality is really dictated by
                                                           #flag_single_number_or_optical_flow_prediction
number_of_neurons_per_layer_time_distributed_model = [5,2]; #Must end with 2 for (shiftx,shifty)
activation_function_per_layer_time_distributed_model = ['linear','linear'];


#IF i'm using a fully convolutional model (optical flow output) i need to make sure no dense layers are allowed:
if flag_single_number_or_optical_flow_prediction == 2:
    flag_put_dense_on_top_of_convolutional_model = 0;
    flag_put_dense_on_top_of_time_distributed_image_model = 0;


#### SINGLE IMAGE MODEL: LAYER SPECIFICATIONS ####
#Save Single Image Model Variables:
exec(save_tic(3));
#
number_of_convolutional_model_layers = 3; #basically proclaimatory
#Get Layer Parameters by specifying each layer parts separately (less compact but probably better):
#INITIALIZE:
kernel_size_list = list();
number_of_filters_list = list();
flag_dilated_convolution_list = list();
flag_resnet_list = list();
flag_batch_normalization_list = list();
flag_size_1_convolution_on_shortcut_in_resnet_list = list();
flag_size_1_convolution_after_2D_convolution_list = list();
flag_batch_normalization_after_size_1_convolution_list = list();
activation_type_str_list = list();
#Layer Specifications:
#Layer 1:
kernel_size_list.append(np.array([7,5,3,1]));
number_of_filters_list.append(np.array([20,20,20,20]));
flag_dilated_convolution_list.append(np.array([0,0,0,0]));
flag_resnet_list.append(np.array([0,0,0,0]));
flag_batch_normalization_list.append(np.array([1,1,1,1]));
flag_size_1_convolution_on_shortcut_in_resnet_list.append(np.array([0,0,0,0]));
flag_size_1_convolution_after_2D_convolution_list.append(np.array([0,0,0,0]));
flag_batch_normalization_after_size_1_convolution_list.append(np.array([1,1,1,1]));
activation_type_str_list.append(['relu','relu','relu','relu']);
#Layer 2:
kernel_size_list.append(np.array([3]));
number_of_filters_list.append(np.array([20]));
flag_dilated_convolution_list.append(np.array([0]));
flag_resnet_list.append(np.array([0]));
flag_batch_normalization_list.append(np.array([1]));
flag_size_1_convolution_on_shortcut_in_resnet_list.append(np.array([0]));
flag_size_1_convolution_after_2D_convolution_list.append(np.array([0]));
flag_batch_normalization_after_size_1_convolution_list.append(np.array([1]));
activation_type_str_list.append(['relu']);
#Layer 3 (one output image/filter, useful for fully-convolutional networks!!!!!!!!!!!!!!!!!!!):
kernel_size_list.append(np.array([3]));
number_of_filters_list.append(np.array([1]));
flag_dilated_convolution_list.append(np.array([0]));
flag_resnet_list.append(np.array([0]));
flag_batch_normalization_list.append(np.array([1]));
flag_size_1_convolution_on_shortcut_in_resnet_list.append(np.array([0]));
flag_size_1_convolution_after_2D_convolution_list.append(np.array([0]));
flag_batch_normalization_after_size_1_convolution_list.append(np.array([1]));
activation_type_str_list.append(['relu']);
#Save Single Image Model Variables:
exec(save_toc(3,3));
#


## ### TIME-DISTRIBUTED IMAGE MODEL (if using optical flow output this MUST end with a single output filter!): ####
#Save Time-Distributed Image Model Variables:
exec(save_tic(4));
#
number_of_convolutional_model_layers_TD = 1; #basically proclaimatory
#INITIALIZE:
kernel_size_list_TD = list();
number_of_filters_list_TD = list();
flag_dilated_convolution_list_TD = list();
flag_resnet_list_TD = list();
flag_batch_normalization_list_TD = list();
flag_size_1_convolution_on_shortcut_in_resnet_list_TD = list();
flag_size_1_convolution_after_2D_convolution_list_TD = list();
flag_batch_normalization_after_size_1_convolution_list_TD = list();
activation_type_str_list_TD = list();
#Layer Specifications:
#Layer 1:
kernel_size_list_TD.append(np.array([3]));
number_of_filters_list_TD.append(np.array([1]));
flag_dilated_convolution_list_TD.append(np.array([0]));
flag_resnet_list_TD.append(np.array([0]));
flag_batch_normalization_list_TD.append(np.array([1]));
flag_size_1_convolution_on_shortcut_in_resnet_list_TD.append(np.array([0]));
flag_size_1_convolution_after_2D_convolution_list_TD.append(np.array([0]));
flag_batch_normalization_after_size_1_convolution_list_TD.append(np.array([1]));
activation_type_str_list_TD.append(['relu']);
#Save Time-Distributed Image Model Variables:
exec(save_toc(4,4));
#




#########################################  ACTUALLY BUILD THE NETWORK #######################################################################
#Build Network:
K.set_learning_phase(1) #training
#(1). Image and correlation Inputs:
#(*). Images:
image_input = Input(shape=(ROI,ROI,number_of_color_channels)); #single image conv model
image_inputs = Input(shape=(number_of_time_steps,ROI,ROI,number_of_color_channels)); #time distributed model
#(*). Cross-Correlations:
if flag_use_discrete_CC_as_input == 1:
    discrete_CC_input = Input(shape=(CC_size,CC_size,1));
if flag_use_sub_pixel_CC_as_input == 1:
    sub_pixel_CC_input = Input( shape=(sub_pixel_CC_size,sub_pixel_CC_size,1) );
if flag_use_deformation_map_as_input == 1:
    deformation_map_input = Input( shape=(deformation_map_size,deformation_map_size,1) );


####### SINGLE IMAGE: #########
#(1). Single Image Convolutional Model:
conv_model_single_image = image_input;
conv_model_single_image = build_convolutional_network(conv_model_single_image,saved_dictionary3);
#(2). MLP on top of every single image model:
if flag_put_dense_on_top_of_convolutional_model==1:
    conv_model_single_image = mlp_block(conv_model_single_image,number_of_neurons_per_layer_convolutional_model,activation_function_per_layer_convolutional_model);


####### TIME-DISTRIBUTED: #########
#(a). take Conv2D model and actually make it a "Model" according to the functional API
conv_model_single_image_as_model = Model(inputs=[image_input],outputs=[conv_model_single_image])
#(b). make it time distributed or bidirectional(TimeDistributed)
conv_model_time_distributed_as_layer = TimeDistributed(conv_model_single_image_as_model)(image_inputs);
#(c). after TimeDistributed we have a tensor of shape (batch_size,number_of_time_steps,single_model_output)
if flag_put_dense_on_top_of_time_distributed_image_model == 0:
    conv_model_time_distributed_as_layer = build_convolutional_network(conv_model_time_distributed_as_layer,saved_dictionary4);
#End of if condition
if flag_put_dense_on_top_of_time_distributed_image_model == 1:
    conv_model_time_distributed_as_layer = mlp_block(conv_model_time_distributed_as_layer,
                                                     number_of_neurons_per_layer_time_distributed_model,
                                                     activation_function_per_layer_time_distributed_model);
#(d). make the whole thing, after TimeDistributed, a Model according to the functional API:
K.set_learning_phase(1); #learning_phase = {1=training, 0=testing/validating}
conv_model_time_distributed_as_model = Model(inputs=[image_inputs],outputs=[conv_model_time_distributed_as_layer])
conv_model_time_distributed_as_model._uses_learning_phase = True #for learning=True, for testing = False
K.set_learning_phase(1)




#(5). #Callbacks:
#model_TensorBoard_function = TensorBoard()
custom_callback_function = custom_callback();
callbacks_list = [custom_callback_function];

#(6). Compile:
metrics_list = ['mae'];
if flag_use_custom_loss_function == 1:
    #i could probably have used a Lambda or lambda function instead but this is more general
    conv_model_time_distributed_as_model.compile(optimizer=optimizer, loss=custom_loss_function, metrics=metrics_list);
else:
    conv_model_time_distributed_as_model.compile(optimizer=optimizer, loss='mse', metrics=metrics_list);


#######################################################################################################################################################
#Save ALL Model Variables (single + time distributed) for later model comparison:
exec(save_toc(2,2));
######################################################################################################################################################


#Load Saved Keras Model if wanted:
if flag_start_new_training_or_continue_training_previous_model == 2:
    #IF continue training previous model:
    conv_model_time_distributed_as_model = keras.models.load_model(model_name_to_continue_training_from);
    if flag_previous_model_training_options == 1:
        #Only take architecture - so reinitializze model weights just in case:
        reinitialize_model_weights(conv_model_time_distributed_as_model);
#End of if condition


#(7).Visualize Model:
if flag_plot_model == 1:
    keras.utils.plot_model(conv_model_single_image_as_model)
    keras.utils.vis_utils.plot_model(conv_model_single_image_as_model,'single_image_model.png');
    keras.utils.plot_model(conv_model_time_distributed_as_model)
    keras.utils.vis_utils.plot_model(conv_model_time_distributed_as_model,'time_distributed_model.png');
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    SVG(model_to_dot(conv_model_single_image_as_model).create(prog='dot', format='svg'))
    SVG(model_to_dot(conv_model_time_distributed_as_model).create(prog='dot', format='svg'))


#(8).Summarize Model:
if flag_print_model_summary == 1:
    conv_model_single_image_as_model.summary();
    conv_model_time_distributed_as_model.summary();


#Search For Current Model Architecture in JSON file:



if flag_start_new_training_or_continue_training_previous_model == 1 or (flag_start_new_training_or_continue_training_previous_model==2 and flag_previous_model_training_options==1):
    #IF start new training or use previous architecture but still do new training -> Use JSON files to find Model Version:
    #TO DO: what if i'm trying to train an architecture which has already been used? create new version or not?

    #(to be able to keep track of different model architectures which were tried):
    #(1). Get current model architecture using json
    current_json_string = conv_model_time_distributed_as_model.to_json()
    #(2). See what other model architectures have existed before:
    #TO DO: from now on it would not only be json files which start with deepspeckles_timedistributed_version...... , so take care of it
    json_files = search_file(basic_model_name + '*' + 'json');
    number_of_versions_so_far = length(json_files);
    model_version = -1;
    if number_of_versions_so_far == 0:
        #Create first version of TimeDistributed EVER:
        with open(basic_model_name + str(0) + ".json", "w") as json_file_fid:
            json_file_fid.write(current_json_string)
        #Set model version:
        model_version = 0;

    else:
        #Previous Versions TimeDistributed were found, now let's see if any has the same architecture...

        #Find out if current model architecture already exists:
        for json_counter in arange(0,number_of_versions_so_far):
            json_file_read_fid = open(json_files[json_counter],'r');
            json_file_json_string = json_file_read_fid.read();
            if current_json_string == json_file_json_string:
                model_version = json_counter; #model version will equal latest json file with current architecture

        if model_version == -1:
            #If current model architecture doesn't exist - add a new model version
            #(1). Write Model Architecture in JSON file for later comparison:
            model_version = number_of_versions_so_far;
            with open(basic_model_name + '_Version_' + str(model_version) + ".json", "w") as json_file_fid:
                json_file_fid.write(current_json_string)

            currrent_model_name = basic_model_name + '_Version_' + str(model_version) + '_TrainingChange_0_ValidationEpoch_0_WeightInitialization_0';

        else:
            #If current model architecture was found - for now just write a warning but write new version json
            message_box('Current Model Architecture Already Found', 'FILE: ' + basic_model_name + str(model_version), 1);
            #(1). Write Model Architecture in JSON file for later comparison:
            model_version = number_of_versions_so_far + 1;
            with open(basic_model_name + '_Version_' + str(model_version) + ".json", "w") as json_file_fid:
                json_file_fid.write(current_json_string)
            #(2). Update Current Model Name:
            current_model_name = basic_model_name + '_Version_' + str(model_version) + 'TrainingChange_0_ValidationEpoch_0_WeightInitialization_0';

elif flag_start_new_training_or_continue_training_previous_model == 2:
    #Use Previous Model:
    # flag_previous_model_training_options == 2 or 3 !
    # model_name_to_continue_training_from = 'DeepSpeckles_TimeDistributed_Version_29_TrainingChanges_0_ValidationEpoch_2_WeightInitialization_0';
    if flag_previous_model_training_options == 2:
        #Model Architecture was found and i want to continue training that same model.
        #In this case i don't have to do anything because current_model_name and version remain the same and will be loaded to console later;
        1;
    #End if
    elif flag_previous_model_training_options == 3:
        #I'm Adding to previous model training i don't need to write another JSON. all i need is to change model name:
        current_model_name = basic_model_name + '_' + str(model_version) + '_' + 'TrainingChange_' + str(training_change+1) + '_ValidationEpoch_0_WeightInitialization_0';

#END of if training new model condition
################################################################################################################################################################################################################
################################################################################################################################################################################################################



################################################################################################################################################################################################################
#########################################################################################################################################################################################################
######################## ACTUALLY TRAIN: ########################

#Batch Generator:
#Perhapse for fit_generator i need a yield operation, but perhapse i should also have a regular
#generator function which simply returns a batch as a regular function for more functionality
# Parameters
generator_parameters = {'ROI':ROI,
                        'speckle_size':speckle_size ,
                        'epp':epp,
                        'readout_noise':readout_noise ,
                        'background': background,
                        'max_shift':max_shift,
                        'number_of_time_steps':number_of_time_steps,
                        'batch_size':batch_size_training,
                        'flag_single_number_or_optical_flow_prediction':flag_single_number_or_optical_flow_prediction ,
                        'constant_decorrelation_fraction':constant_decorrelation_fraction ,
                        'Fs':Fs,
                        'flag_center_each_speckles_independently_or_as_batch_example':flag_center_each_speckles_independently_or_as_batch_example ,
                        'flag_stretch_each_speckles_independently_or_as_batch_example':flag_stretch_each_speckles_independently_or_as_batch_example ,
                        'std_wanted':std_wanted,
                        'flag_stateful_batch':flag_stateful_batch ,
                        'flag_data_generation_online_or_from_file':flag_data_generation_online_or_from_file,
                        'file_name_X': file_name_to_get_training_data_from,
                        'simulation_parameters_klepto_file_name': simulation_parameters_klepto_file_name,
                        'flag_shuffle':flag_shuffle,
                        'flag_get_data_as_function_or_as_generator': flag_get_data_as_function_or_as_generator}

generator_parameters_validation = dict(generator_parameters);
generator_parameters_validation['batch_size'] = batch_size_validation;
generator_parameters_validation['flag_data_generation_online_or_from_file'] = 1; #for validation always use as function
#Generator objects:
training_generator_object = DataGenerator(**generator_parameters)
validation_generator_object = DataGenerator(**generator_parameters_validation)
#Generator:
if flag_get_data_as_function_or_as_generator == 2:
    training_generator = training_generator_object.generate_iterator(number_of_batches_to_load_to_RAM);
    validation_generator = validation_generator_object.generate_iterator(1);


###tensorflow session (TO DO: incorporate tensorflow sessions with keras to be able to incorporate
#                      tensorflow stuff like TFRecords into tensorflow):
#sess = tf.Session()
#K.set_session(sess)
#init_op = tf.global_variables_initializer()
#sess.run(init_op)


#Actually Fit:
training_error_over_time = np.empty(shape=(1,));
validation_error_over_time = np.empty(shape=(1,));

#Get constant validation set:
validation_X,validation_y,flag_validation_EOF = validation_generator_object.generate_function(number_of_validation_batches_to_check_with);
#show_matrices_video(validation_X,0.4,5,1,1,1,1)
validation_data_predictions = np.ndarray(shape=shape(validation_y));



##################################################################################################################################
#Experimentation
#TO DO: save results matrix probably in a seperate script because even if i'm training previous model
#but only using architecture it's still relevant to initialize results_matrix. in general i should probably
#update results_matrix klepto file in accordance with the situation.
#(1). new training - initialize to zero and create new klepto
#(2). previous model + only architecture - initialize to zero and create new klepto (new version of previous model)
#(3). previous model + train to completion - load klepto and that's it
#(4). previous model + add batches - load klepto and reinitialize a results_matrix with more number_of_data_generations
#(*). Hyper-Parameter Tracking:
hyper_parameters_list = ['batch_size_training','learning_rate','dropout_rate','flag_stateful_batch','weight_initialization_counter','momentum','optimizer',
                         'flag_changed_samples_origin'];
#(*). Results List:
#i want to keep track of the different training/validation results of current model version through all the different phases.
#for that i have a results list of lists. in order to be able to quickly visually see the different training results i will need to build an
#auxiliary functions bank to be able to quickly manipulate this list to visually see what i want.
#the results matrix will be built with dimensions as such:
#(0). Different Weight Initializations - the thing is...this is not the current script weight_counter but this must be an internal counter to keep
#                                        track of total number of times different weight initializations were tried
#(1). Different Hyper-Parameters / Training-Examples - why designate different training examples as a meaningful change? because what if i pretrained my network
#                                                      on certain samples (non-turbulent samples for instance) and i want to continue training on different
#                                                      samples (turbulent samples for instance) this could be a meaningful event in the history of my network
#(2). Different Validation Epochs
#(3). Training losses/metrics for current validation epoch
#(4). Validation losses/metrics of current validation epoch
#(****). The thing is....what if i'm trying to fidel around with the network without it being "formal" .... should everytime i fidel around with hyperparameters
#        enter to the "formal" results list? and how do i keep track of myself fideling around within the script? maybe i should have a block right before
#        or within the training loop which checks if a hyper-parameter was changed.
#        I should probably also have a flag_update_results_list to work around it if i'm just playing around with the script
results_matrix = zeros( (number_of_different_weights_initializations, number_of_data_generations_or_loads_to_memory ) );
results_list = list() #Weight Initializations
results_list.append('temporary dummy variable to initialize zero index'); #dummy variable - add spots for different weight initializations
results_list[0] = list(); #Hyper-Parameters / Training-Examples
results_list[0].append('temporary dummy variables to initialize zero index'); #for weight initialization zero add spots for hyper-parameters / training-examples
results_list[0][0] = list() #Validation-Epoch
results_list[0][0].append(list()); #a list for training results
results_list[0][0].append(list()); #a list for validation results
results_list[0][0][0].append([1,2])
results_list[0][0][1].append([3,4])
#Tracking counters/variables:
weight_initialization_index = 0;
training_change_index = 0;
results_list_size = shape(results_list)
results_list_weight_initialization_size = results_list_size[0];
results_list_training_change_size = results_list_size[1];
results_list_number_of_metrics_kept_track_of = results_list_size[2];
results_list_specific_training_phase_validation_size = size(results_list[weight_initialization_index][training_change_index][1])
#Examples of how to add current epoch results using tracking variables
current_epoch_training_losses = numpy.random.randn(10,1);
current_epoch_validation_loss = 4;
results_list[weight_initialization_index][training_change_index][0].append(current_epoch_training_losses)
results_list[weight_initialization_index][training_change_index][1].append(current_epoch_validation_loss);
#Example of how to start a new weight initialization to the training:
new_weight_initialization_member_in_results_list = list(); #it is itself a list of training changes
new_weight_initialization_member_in_results_list.append('dummy variables');
new_weight_initialization_member_in_results_list[0] = list(); #it is itself a list of different metrics kept track of
new_weight_initialization_member_in_results_list[0].append(list()); #a list with training losses
new_weight_inttialization_member_in_results_list[0].append(list()); #a list with validation losses
new_weight_initialization_member_in_results_list[0][0].append(current_epoch_training_losses);
new_weight_initialization_member_in_results_list[0][1].append(current_epoch_validation_loss);
#Example of how to start a new training change initialization to the training:
new_training_change_member_in_results_list = list(); #it is itself a list of metrics kept track of
new_training_change_member_in_results_list.append(list()); #a list of training losses
new_training_change_member_in_results_list.append(list()); #a list of validation losses
new_training_change_member_in_results_list[0].append(current_epoch_training_losses);
new_training_change_member_in_results_list[1].append(current_epoch_validation_losses);


batch_counter = 1;
validation_counter = 1;

##################################################################################################################################
#Save current session variables:
exec(save_toc(1,1));
#Save to whole script klepto:
#DO: understand whether all the variables in saved_variables1 should be saved in accordance with the different flags
#
if flag_start_new_training_or_continue_training_previous_model == 1 or \
    (flag_start_new_training_or_continue_training_previous_model==2 and flag_previous_model_training_options!=2):
    #TO DO: change name to be more complete (include version and for example)
    file_name_whole_script_klepto = basic_model_name + '_Version_' + str(model_version) + '_whole_script_klepto'
    whole_script_klepto_db = save_variables_to_klepto_file(file_name_whole_script_klepto,saved_dictionary1);
##################################################################################################################################


#Load previous model to continue training if wanted (THIS MEANS MODEL BUILDING BEFORE DOESN'T MEAN ANYTHING):

#TO DO:
#i'f im adding training to an older model weight initialization or something like that i should skip weight initialization....
#but i need to make sure it works and maybe add a "different_weihts_initializations_stop" variable??????????????????
if flag_start_new_training_or_continue_training_previous_model == 2 and flag_previous_model_training_options!=1:
    #IF i'm continuing to train an old model:
    exec(load_variables_from_klepto_file(whole_script_klepto_file_name));
    different_weights_initializations_start = different_weights_counter;
    data_generations_or_loads_start = data_loads_counter;
    flag_skip_weight_initialization = 1;
else:
    #Starting to train a new model:
    different_weights_initializations_start = 0;
    data_generations_or_loads_start = 0;
    flag_skip_weight_initialization = 0;




#START TRAINING LOOP:
global_fit_function_counter = 0;
global_validation_counter = 0;
try:

    if flag_method_of_fitting == 1:
        #(1). Fit using large data already in the form of numpy arrays

        #I SHOULD ALSO CHANGE THE LOOP and also add: different_weights_initializatioins_stop !@!#!#!
        for different_weights_counter in arange(different_weights_initializations_start,number_of_different_weights_initializations,1):

            tic_toc.tic()
            if flag_skip_weight_initialization == 0:
                #Reset Model Weights to try learning from another weight instansiation:
                reinitialize_model_weights(conv_model_single_image_as_model);
                reinitialize_model_weights(conv_model_time_distributed_as_model);

                #Reinitialize training and validation errors over time vector to keep track of current model weights initializations:
                training_error_over_time = np.empty(shape=(1,));
                validation_error_over_time = np.empty(shape=(1,));

            elif flag_skip_weight_initialization == 1:
                1;
                #flag_skip_weight_initialization = 0;

            print('********************')
            print('Reinitialize Weights');
            print('*********************')
            tic_toc.toc(True);

            #Try to fit the model:
            for data_loads_counter in arange(data_generations_or_loads_start,number_of_data_generations_or_loads_to_memory):

                #Generate batches for training:
                #when flag_EOF is reached this means that
                tic_toc.tic()
                [X,y,flag_EOF] = training_generator_object.generate_function(number_of_batches_to_load_to_RAM);
                print('Generate {X,y}');
                tic_toc.toc(True);

                #epochs_per_generated_training_data will almost always be 1 unless i'm trying on purpose
                #to overfit in order to see if the model can learn.
                for epochs_over_current_generated_data_counter in arange(0,epochs_per_generated_training_data):
                #Fit:

                    #Parameters to track indices of examples within batches loaded to RAM:
                    fit_function_counter = 0;
                    start_index = 0;
                    stop_index = 0;

                    #Start going through batches and train:
                    while stop_index <= number_of_batches_to_load_to_RAM:
                        #Get proper indices:
                        start_index = batch_size_training*number_of_batches_to_give_to_fit_function*fit_function_counter
                        stop_index = start_index + batch_size_training*number_of_batches_to_give_to_fit_function
                        indices = arange(start_index,stop_index);
                        fit_function_counter = fit_function_counter + 1;

                        #Check we havn't reached EOF so that indices could be out of range!!!!:
                        if flag_EOF == 1:
                            print('Reached EOF');
                            1;

                        #Fit function on current samples/batches:
                        tic_toc.tic()
                        history = conv_model_time_distributed_as_model.fit(x=X[indices,:], y=y[indices,:],
                                                                      batch_size = batch_size_training,
                                                                      callbacks = callbacks_list);
                        print('.fit() function')
                        tic_toc.toc(True);

                        #Accumulate Traning Error Over Time:
                        current_fit_function_training_error_over_time = np.asarray(custom_callback_function.error_over_time_training);
                        custom_callback_function.error_over_time_training = []; #simply to explicitly reinitialize
                        training_error_over_time = np.concatenate((training_error_over_time,current_fit_function_training_error_over_time))

                        #Update results matrix:
                        #Add possibility of appending according to training additions / changes etc'. probably use lists
                        #(*) Perhaps i would like to only continue training a certain weight initialization...what do i do then?
                        #(*) Maybe i should have results_matrix_training NOT be created a new when i continue training a certain
                        #    model (and certain weight initialization) but continue updating the original results_matrix????....

                        #The results matrix possibilities:
                        #1. Using Panda as saving the table
                        results_matrix_training[different_weights_counter,global_fit_function_counter,:] =\
                            current_fit_function_training_error_over_time;
                        global_fit_function_counter = global_fit_function_counter + 1;

                        #Validation if it's time:
                        batch_counter = batch_counter + number_of_batches_to_give_to_fit_function;
                        if batch_counter > number_of_batches_before_validation:
                            batch_counter = 0;
                            validation_counter = validation_counter + 1;
                            print('Validation Mode:');

                            tic_toc.tic();
                            #Get validation data predictions:
                            K.set_learning_phase(0);
                            validation_data_predictions = conv_model_time_distributed_as_model.predict(validation_X,batch_size=1)
                            K.set_learning_phase(1);
                            print('Model validation prediction')
                            tic_toc.toc(True);


                            tic_toc.tic();
                            #Rename for easy use:
                            prediction_x = validation_data_predictions[:,0];
                            prediction_y = validation_data_predictions[:,1];
                            label_x = validation_y[:,0];
                            label_y = validation_y[:,1];

                            #Calculate error over time validation:
                            current_validation_loss = 1/2 * 1/length(label_x) * (sum((prediction_x-label_x)**2) + sum((prediction_y-label_y)**2))
                            validation_error_over_time = np.concatenate( (validation_error_over_time, np.atleast_1d(current_validation_loss)));
                            print('Get error over time')
                            tic_toc.toc(True);

                            #Update results matrices for validation errors tracking:
                            results_matrix_validation[different_weights_counter,global_validation_counter,:] = current_validation_loss;
                            global_validation_counter = global_validation_counter + 1;


                            ############################################################################################################
                            tic_toc.tic()
                            #Save Model (Architecture + ) Weights (Every validation step):
                            current_model_name_full = basic_model_name  + \
                                                               '_Version_' + str(model_version) + \
                                                               '_ValidationEpoch_' + str(validation_counter) + \
                                                               '_WeightInitializations_' + str(different_weights_counter) + \
                                                               '.h5';
                            conv_model_time_distributed_as_model.save(current_model_name_full)
                            # conv_model_single_image_as_model.save(current_model_name_full)
                            print('Save model');
                            tic_toc.toc(True);
                            ############################################################################################################



                            ############################################################################################################
                            #Update Klepto file tracking training:

                            ############################################################################################################

                            tic_toc.tic();
                            ############################################################################################################
                            # MATPLOTLIB.PLT CONTAINED APPROACH - COULD AND SHOULD BE MUCH MORE EFFICIENT AND INTERACTIVE !!! ####################################
                            plt.close(1);
                            figure(1);
                            subplot(2,1,1);
                            plot(validation_data_predictions[:,0],'b');
                            plot(validation_y[:,0],'g')
                            legend(('prediction','actual'))
                            title('validation data vs. predictions (X-axis)')

                            subplot(2,1,2);
                            plot(validation_data_predictions[:,1]);
                            plot(validation_y[:,1],'g')
                            legend(('prediction','actual'))
                            title('validation data vs. predictions (Y-axis)')
                            plt.show();

                            #Plot what's going on over time:
                            #(1). Scale both x-axes to the same time scale:
                            x_axis_training = arange(1,length(training_error_over_time)+1);
                            x_axis_validation = arange(1,length(validation_error_over_time)+1)*number_of_batches_before_validation;
                            #(2). Plot:
                            plt.close(2);
                            figure(2);
                            plot(x_axis_training[1:],10*log10(training_error_over_time[1:]),'b');
                            ylabel('training_loss (blue) & validation_loss (green)');
                            xlabel('batches');
                            title('training and validation loss');
                            #Add validation to the same plot
                            plot(x_axis_validation[1:],10*log10(validation_error_over_time[1:]),'g');
                            plt.show();

                            #Plot current validation epoch error over time to look at finer details
                            #at what's going on:
                            plt.close(3);
                            figure(3);
                            plot(current_fit_function_training_error_over_time);
                            ylabel('training error [linear]');
                            xlabel('latest batches');
                            title('Current Validation Epoch Training Error Over Time')
                            plt.show();

                            #Pause to give matplotlib time to work:
                            plt.close(4);
                            figure(4);
                            plt.show();
                            # time.sleep(1);
                            print('plotting');
                            tic_toc.toc(True);
                            ############################################################################################################



                            ############################################################################################################
                            #DECIDE ON LEARNING RATE:
                            if flag_use_learning_rate_schedualer == 1:

                                #Is current score significantly different from the best so far?:
                                if tolerance_around_best_score_linear < abs(best_validation_loss_so_far/current_validation_loss - 1):
                                    #New Best Score Found:
                                    best_validation_loss_so_far = current_validation_loss
                                    counter_to_learning_rate_update = 0
                                    if verbose > 0:
                                        print('---current best val accuracy: %.3f' % current_validation_loss)

                                else:
                                    #Score worse or insignificant relative to Best Score:
                                    if counter_to_learning_rate_update >= times_to_wait_before_learning_rate_update:
                                        #I've waited enough time to justify changing learning rate:

                                        #Is it time to stop learning?:
                                        counter_number_of_times_learning_rate_was_reducted += 1
                                        if counter_number_of_times_learning_rate_was_reducted <= number_of_learning_rate_reductions_before_learning_stops:
                                            #No - reduce learning rate:
                                            lr = K.get_value(conv_model_time_distributed_as_model.optimizer.lr)
                                            K.set_value(conv_model_time_distributed_as_model.optimizer.lr, lr*learning_rate_reduction_factor)
                                            print('Learning rate reduced');
                                        else:
                                            #Yes - we have had enough learning rate reductions:
                                            if verbose > 0:
                                                print("Stopped Learning - there were enough learning rate reductions");
                                            conv_model_time_distributed_as_model.stop_training = True

                                    #Update counter keeping track of how many times score wasn't improved:
                                    counter_to_learning_rate_update += 1
                            ############################################################################################################

                        #End of if validation condition

                    #End of RAM counter
            #END of number of epochs loop
        #END of number of weight reinitializations loop

    elif flag_method_of_fitting == 2:
        #(2). Fit using a Generator function:
        number_of_workers = 1;
    #    K.set_learning_phase(0)
        conv_model_time_distributed_as_model.fit_generator(generator = training_generator,
                                                  steps_per_epoch = 100,
                                                  validation_data = validation_generator,
                                                  validation_steps = 10,
                                                  workers = number_of_workers)
    elif flag_method_of_fitting == 3:
        #(3). Fitting using one fit per loop (more controllable):
        1;

except KeyboardInterrupt:

        #Update Klepto file tracking training:
        whole_script_klepto_db['best_validation_loss_so_far'] = best_validation_loss_so_far;
        whole_script_klepto_db['current_validation_loss'] = current_validation_loss;
        whole_script_klepto_db['counter_to_learning_rate_update'] = counter_to_learning_rate_update;
        whole_script_klepto_db['counter_number_of_times_learning_rate_was_reducted'] = counter_number_of_times_learning_rate_was_reducted;
        whole_script_klepto_db['lr'] = lr;
        whole_script_klepto_db['data_loads_counter'] = data_loads_counter;
        whole_script_klepto_db['different_weights_counter'] = different_weights_counter;
        whole_script_klepto_db['start_index'] = start_index;
        whole_script_klepto_db['stop_index'] = stop_index;
        whole_script_klepto_db['fit_function_counter'] = fit_function_counter;
        whole_script_klepto_db['training_error_over_time'] = training_error_over_time;
        whole_script_klepto_db['validation_error_over_time'] = validation_error_over_time;
        whole_script_klepto_db['results_matrix_training'] = results_matrix_training;
        whole_script_klepto_db['results_matrix_validation'] = results_matrix_validation;

        print("press control-c again to quit")



#### KERAS MODEL FUNCTIONS: ####
#Model.build
#Model.compile
#Model.compute_output_shape
#Model.count_params
#Model.evaluate
#Model.fit
#Model.get_layer
#Model.get_weights
#Model.history
#Model.predict
#Model.save
#Model.save_weights
#Model.stateful
#Model.summary
#Model.to_json
#Model.to_yaml
#Model.weights



###########################  PYQTGRAPH  ##########################################################################
# #(1). pyqtgraph app:
# pyqtgraph_app = QtGui.QApplication([])
# #(2). graphics window:
# #(*)
# gui_graphics_window_validation = pg.GraphicsWindow(title="Validation graphs: predcitions vs. labels")
# gui_graphics_window_validation.resize(1000,600)
# #gui_graphics_window_validation.setWindowTitle('pyqtgraph example: Plotting'); #THIS IS HOW I CHANGE WIINDOW LABEL
# #(*)
# gui_graphics_window_errors = pg.GraphicsWindow(title="Training and Validation Error over time")
# gui_graphics_window_errors.resize(1000,600)
# #(3). actual plots themselves:
# #(*)
# validation_x_plot = gui_graphics_window_validation.addPlot(title="Validation: prediction X vs. Real X")
# validation_x_plot.setLabel('left', "Prediction", units='pixels')
# validation_x_plot.setLabel('bottom', "Time")
# validation_x_plot.setLogMode(x=False, y=False); #maybe later for large amount of data
# #(*)
# validation_y_plot = gui_graphics_window_validation.addPlot(title="Validation: prediction Y vs. Real Y")
# validation_y_plot.setLabel('left', "Prediction", units='pixels')
# validation_y_plot.setLabel('bottom', "Time")
# validation_y_plot.setLogMode(x=False, y=False); #maybe later for large amount of data
# #(*)
# errors_over_time_plot = gui_graphics_window_errors.addPlot(title="Training and Validation Errors over time")
# errors_over_time_plot.setLabel('left', "Errors", units='pixels')
# errors_over_time_plot.setLabel('bottom', "Batches")
# # go down a row and present current "epoch" (number of steps before validation) training losses:
# gui_graphics_window_validation.show();
# gui_graphics_window_errors.show();
# pg.show();
# pause(1);
##################################################################################################################################




# ############## PYQTGRAPH PLOTS ###############
# #Clear plots before plotting again:
# validation_x_plot.clear();
# validation_y_plot.clear();
# errors_over_time_plot.clear();
#
# #Plot Validation data prediction results:
# #(1). X axis:
# validation_x_plot.plot(prediction_x, pen=mkPen(color='b', width = 1));
# validation_x_plot.plot(label_x, pen=mkPen(color='g', width = 1));
# #(2). Y axis:
# validation_y_plot.plot(prediction_y, pen=mkPen(color='b', width = 1));
# validation_y_plot.plot(label_y, pen=mkPen(color='g', width = 1));
#
# #Plot Training and Validation Errors Over Time:
# training_x_axis = arange(0,length(training_error_over_time));
# validation_x_axis = arange(0,length(validation_error_over_time));
# validation_x_axis = validation_x_axis * number_of_batches_before_validation;
# errors_over_time_plot.plot(training_x_axis, training_error_over_time, pen=mkPen(color='b', width = 1));
# errors_over_time_plot.plot(validation_x_axis, validation_error_over_time, pen=mkPen(color='g', width=1));
# pause(1);
# gui_graphics_window_validation.show();
# gui_graphics_window_errors.show();
# pause(1);
# ##########################################################################





#figure(1);
#                    plot(validation_data_predictions[:,0],'b');
#                    plot(validation_y[:,0],'g')
#                    legend(('prediction','actual'))
#                    title('validation data vs. predictions (Y-axis)')
#                    figure(2);
#                    plot(validation_data_predictions[:,1]);
#                    plot(validation_y[:,1])
#                    legend(('prediction','actual'))
#                    title('validation data vs. predictions (X-axis)')
#
#                    #Plot what's going on:
#                    figure(3);
#                    plot(10*log10(training_error_over_time[1:]));
#                    ylabel('val_loss');
#                    xlabel('epochs');
#                    title('training loss');
#                    figure(4);
#                    plot(10*log10(validation_error_over_time[1:]));
#                    ylabel('val_loss');
#                    xlabel('epochs');
#                    title('validation loss');

################################################################################################################################################################################################################






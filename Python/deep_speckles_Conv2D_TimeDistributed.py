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
import os
import datetime
import time #time.time()
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
length = len #use length instead of len.... make sure it doesn't cause problems

from numpy import power as power
from numpy import exp as exp
from pylab import imshow, pause, draw, title, axes, ylabel, ylim, yticks, xlabel, xlim, xticks
from pylab import colorbar, colormaps, colors, subplot, suptitle, plot



#keras.utils.vis_utils.plot_model
######################################################################################################################################################################################################v
######################################################################################################################################################################################################
########################################################################################################################################################################################################################################################################
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



######### Math/Random Functions/Constants: #########
from math import sin
from math import pi
#from math import exp
from math import pow
from random import random, randint, uniform, randrange, choice, gauss, shuffle


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


from importlib import reload
import things_to_import
#things_to_import = reload(things_to_import)
#from things_to_import import *


#import int_range, int_arange, mat_range, matlab_arange, my_linspace, my_linspace_int,importlib
#from int_range import *
#from int_arange import *
#from mat_range import * 
#from matlab_arange import *
#from my_linspace import *
#from my_linspace_int import *
import get_center_number_of_pixels
get_center_number_of_pixels = reload(get_center_number_of_pixels);
from get_center_number_of_pixels import *
import get_speckle_sequences
get_speckle_sequences = reload(get_speckle_sequences);
from get_speckle_sequences import *

####################################################################################################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################




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
def mlp_block(input_layer, number_of_neurons_per_layer):    
    number_of_layers = len(np.array(number_of_neurons_per_layer));
    input_layer = Flatten()(input_layer);
    mlp_block = Dense(number_of_neurons_per_layer[0])(input_layer);
    for layer_counter in arange(1,number_of_layers-1,1):
        mlp_block = Dense(number_of_neurons_per_layer)(mlp_block);
    
    return mlp_block
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
               max_shift = 0.1, 
               number_of_time_steps = 4, 
               batch_size = 32, \
               flag_single_number_or_optical_flow_prediction = 1, \
               constant_decorrelation_fraction = 0, 
               Fs = 5500,\
               flag_center_each_speckles_independently_or_as_batch = 2, \
               flag_stretch_each_speckles_independently_or_as_batch = 2, \
               std_wanted = 1, \
               flag_stateful_batch = 1, \
               flag_online_or_from_file = 1,
               file_name = 'bla',
               flag_shuffle = False):
      
      #If i only have a function to generate one sample, and i write down in the model.fit function a certain
      #batch size...will it automatically generate the needed number of batch samples?.
      
      #Is it possible to "build up" in-batch calculations and they say "fit" on calculations so far?
      
      #It seems very memory inefficient to load to memory a "sample" a few images, and then, whilst within(!)
      #the same batch, with the same hidden states without reseting, to load a new "sample" which only has one 
      #image different!!!. ASK DEEP LEARNING ISRAEL FOR HELP!!@!#!!@$^%
      
      #if i use fit_generator is has a variable "shuffle=true" as default!@#!#!
      
      #As it is now i cannot have a within-batch reseting of gates, i must have the entire batch be of 
      #the same hidden state memory.
      

      #(1). Macro Parameters:
      # 'Initialization'
      self.ROI = ROI;
      self.number_of_time_steps = number_of_time_steps;
      self.batch_size = batch_size;
      self.flag_shuffle = flag_shuffle; #i don't see any reason for this to be true as i do this mehanically
      self.flag_single_number_or_optical_flow_prediction = flag_single_number_or_optical_flow_prediction;
      
      #(2). Choose online data generation or from file
      self.flag_online_or_from_file = flag_online_or_from_file; #can i even add such a variable????
      self.file_name = file_name;
      
      #(3). Speckles Sequence Simulation parameters:
      self.constant_decorrelation_fraction = constant_decorrelation_fraction; #this is mostly important for turbulence
      self.Fs = Fs
      
      #(4). Specific Speckles parameters:
      self.speckle_size = speckle_size;
      self.epp = epp
      self.readout_noise = 140;
      self.max_shift = 0.1
      self.flag_center_each_speckles_independently_or_as_batch = flag_center_each_speckles_independently_or_as_batch
      self.flag_stretch_each_speckles_independently_or_as_batch = flag_stretch_each_speckles_independently_or_as_batch
      self.std_wanted = std_wanted
      self.flag_stateful_batch = flag_stateful_batch;

      
      
  def generate_iterator(self,number_of_batches):
      'Generates batches of samples'
      # Infinite loop
      while 1==1:
          
          #Initialize Variables:
          #(1). Image Sequence:
          X = np.empty((int(self.batch_size), int(self.number_of_time_steps), int(self.ROI), int(self.ROI), 1)); #Images
          #(2). Labels:
          if self.flag_single_number_or_optical_flow_prediction == 1:
              y = np.empty((int(self.batch_size),2)); #Shift {x&y} between frame N/2 and N/2+1
          else:
              y = np.empty((int(self.batch_size),int(self.ROI),int(self.ROI),2)); #Optical flow (2 image sized predictions, one for x and one for y)
          
            
          #Decide Whether online or from file:
          if self.flag_online_or_from_file==1:
             #Online
             ROI = self.ROI
             speckle_size = self.speckle_size
             epp = self.epp
             readout_noise = self.readout_noise
             max_shift = self.max_shift
             number_of_time_steps = self.number_of_time_steps
             batch_size = self.batch_size
             flag_single_number_or_optical_flow_prediction = self.flag_single_number_or_optical_flow_prediction
             constant_decorrelation_fraction = self.constant_decorrelation_fraction
             Fs = self.Fs
             flag_center_each_speckles_independently_or_as_batch = self.flag_center_each_speckles_independently_or_as_batch
             flag_stretch_each_speckles_independently_or_as_batch = self.flag_stretch_each_speckles_independently_or_as_batch
             flag_stateful_batch = self.flag_stateful_batch;
                

             for batch_counter in arange(0,number_of_batches,1):
                 [X_current,y_current] = get_speckle_sequences(ROI, 
                                               speckle_size, 
                                               epp, 
                                               readout_noise, 
                                               max_shift, 
                                               number_of_time_steps, 
                                               batch_size, \
                                               flag_single_number_or_optical_flow_prediction, \
                                               constant_decorrelation_fraction, 
                                               Fs,\
                                               flag_center_each_speckles_independently_or_as_batch, \
                                               flag_stretch_each_speckles_independently_or_as_batch, \
                                               std_wanted, \
                                               flag_stateful_batch);
                 X = np.concatenate([X,X_current],axis=0);
                 y = np.concatenate([y,y_current],axis=0);
             yield X,y
             
          else:
             #From file (load from binary or np or something - check it out!!!)
             yield 1,1
             
          #END of if flag_online_or_from_file==1
      #END of while 1:     
  #END of generator function in DataGenerator class
  
  def generate_function(self,number_of_batches):
      'Generates batches of samples'
      # Infinite loop
      if 1==1: #Notice the change from while to if to keep it consistent with the iterator
          
          #Initialize Variables:
          #(1). Image Sequence:
          X = np.empty((int(self.batch_size), int(self.number_of_time_steps), int(self.ROI), int(self.ROI), 1)); #Images
          #(2). Labels:
          if self.flag_single_number_or_optical_flow_prediction == 1:
              y = np.empty((int(self.batch_size),2)); #Shift {x&y} between frame N/2 and N/2+1
          else:
              y = np.empty((int(self.batch_size),int(self.ROI),int(self.ROI),2)); #Optical flow (2 image sized predictions, one for x and one for y)
          
            
          #Decide Whether online or from file:
          if self.flag_online_or_from_file==1:
             #Online
             ROI = self.ROI
             speckle_size = self.speckle_size
             epp = self.epp
             readout_noise = self.readout_noise
             max_shift = self.max_shift
             number_of_time_steps = self.number_of_time_steps
             batch_size = self.batch_size
             flag_single_number_or_optical_flow_prediction = self.flag_single_number_or_optical_flow_prediction
             constant_decorrelation_fraction = self.constant_decorrelation_fraction
             Fs = self.Fs
             flag_center_each_speckles_independently_or_as_batch = self.flag_center_each_speckles_independently_or_as_batch
             flag_stretch_each_speckles_independently_or_as_batch = self.flag_stretch_each_speckles_independently_or_as_batch
             flag_stateful_batch = self.flag_stateful_batch;
                

             for batch_counter in arange(0,number_of_batches,1):
                 [X_current,y_current] = get_speckle_sequences(ROI, 
                                               speckle_size, 
                                               epp, 
                                               readout_noise, 
                                               max_shift, 
                                               number_of_time_steps, 
                                               batch_size, \
                                               flag_single_number_or_optical_flow_prediction, \
                                               constant_decorrelation_fraction, 
                                               Fs,\
                                               flag_center_each_speckles_independently_or_as_batch, \
                                               flag_stretch_each_speckles_independently_or_as_batch, \
                                               std_wanted, \
                                               flag_stateful_batch);
                 X = np.concatenate([X,X_current],axis=0);
                 y = np.concatenate([y,y_current],axis=0);
             return X,y
             
          else:
             #From file (load from binary or np or something - check it out!!!)
             return 1,1
             
          #END of if flag_online_or_from_file==1
      #END of while 1:     
  #END of generator function in DataGenerator class
  
#END OF DataGenerator:
     
        
################################################################################################################################################################################################################
################################################################################################################################################################################################################
###############################################################################################################################################################################################################

############ CALLBACKS: #############

#ModelCheckpoint(filepath,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto',period=1);
#####   filepath: weights.{epoch:02d}_{val_loss:.2f}.hdf5
#EarlyStopping(monitor='val_loss',min_delta=0,patience=0,vrbose=0,mode='auto');
#ReduceOnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=0,mode='auto',epsilon=0.001,cool_down=0,min_lr=0);
#LambdaCallback(on_epoch_begin=None,on_epoch_end=None,on_batch_begin=None,on_batch_end=None,on_train_begin=None,on_train_end=None)
#TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, 
#            write_grads=False, write_images=False, 
#            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#### if you have installed tensorflow with pp you should be able to launce tensorboard from the command line:
#### tensorboard --logdir=/full_path_to_your_logs

class custom_callback(keras.callbacks.Callback):
    # logs include: loss, acc, val_loss, val_acc
        
    #Train:
    def on_train_begin(self,logs={}):
        #Initialize class variables:
        self.error_over_time = []; 
    def on_train_end(self,logs={}):
        1;
        
    #Batch:
    def on_batch_begin(self,batch,logs={}):
        1;
    def on_batch_end(self,batch,logs={}):
        1;
    
    #Epoch:
    def on_epoch_begin(self,epoch,logs={}):
        1;
    def on_epoch_end(self,epoch,logs={}):
        #self.error_over_time.append(custom_validation_function());   
        1;

################################################################################################################################################################################################################
################################################################################################################################################################################################################
#### USING TIME DISTRIBUTED LAYER - DOING IT RIGHT UNLIKE WHAT I DID ABOVE: ######


################################################################################################################################################################################################################
########################## BUILD THE NETWORKS: ####################


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

#Convolutional Model:
number_of_convolutional_model_layers = 3; #~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Layer 1:
kernel_size_list.append(np.array([7,5,3,1]));
number_of_filters_list.append(np.array([4,4,4,4]));
flag_dilated_convolution_list.append(np.array([0,0,0,0]));
flag_resnet_list.append(np.array([0,0,0,0]));
flag_batch_normalization_list.append(np.array([1,1,1,1]));
flag_size_1_convolution_on_shortcut_in_resnet_list.append(np.array([0,0,0,0]));
flag_size_1_convolution_after_2D_convolution_list.append(np.array([0,0,0,0]));
flag_batch_normalization_after_size_1_convolution_list.append(np.array([1,1,1,1]));
activation_type_str_list.append(['relu','relu','relu','relu']); 
#Layer 2:
kernel_size_list.append(np.array([3]));
number_of_filters_list.append(np.array([5]));
flag_dilated_convolution_list.append(np.array([0]));
flag_resnet_list.append(np.array([0]));
flag_batch_normalization_list.append(np.array([1]));
flag_size_1_convolution_on_shortcut_in_resnet_list.append(np.array([0]));
flag_size_1_convolution_after_2D_convolution_list.append(np.array([0]));
flag_batch_normalization_after_size_1_convolution_list.append(np.array([1]));
activation_type_str_list.append(['relu']);  
#Layer 3 (one output image/filter, useful for fully-convolutional networks):
kernel_size_list.append(np.array([3]));
number_of_filters_list.append(np.array([1]));
flag_dilated_convolution_list.append(np.array([0]));
flag_resnet_list.append(np.array([0]));
flag_batch_normalization_list.append(np.array([1]));
flag_size_1_convolution_on_shortcut_in_resnet_list.append(np.array([0]));
flag_size_1_convolution_after_2D_convolution_list.append(np.array([0]));
flag_batch_normalization_after_size_1_convolution_list.append(np.array([1]));
activation_type_str_list.append(['relu']);  


#Auxiliary parameters:
ROI_placeholder = K.placeholder(shape=(None));
number_of_time_steps_placeholder = K.placeholder(shape=(None));
input_size = (ROI_placeholder*ROI_placeholder,2);

##############################################################################
##############################################################################
##############################################################################
#Network parameters:
flag_plot_model = 0;
ROI = 32;
number_of_time_steps = 4;
CC_size = 3;
##############################################################################
##############################################################################
##############################################################################


                                     
#Build Network:
#(1). Image Input:
number_of_color_channels = 1;
image_input = Input(shape=(ROI,ROI,number_of_color_channels));
image_inputs = Input(shape=(number_of_time_steps,ROI,ROI,number_of_color_channels)); #????????

conv_model_single_image = image_input;
for layer_counter in arange(0,number_of_convolutional_model_layers,1):
    number_of_filters_in_layer = number_of_filters_list[layer_counter];
    kernel_size_in_layer = kernel_size_list[layer_counter];
    flag_dilated_convolution_in_layer = flag_dilated_convolution_list[layer_counter];
    flag_resnet_in_layer = flag_resnet_list[layer_counter];
    flag_size_1_convolution_on_shortcut_in_resnet_in_layer = flag_size_1_convolution_on_shortcut_in_resnet_list[layer_counter];
    flag_batch_normalization_in_layer = flag_batch_normalization_list[layer_counter];
    flag_size_1_convolution_after_2D_convolution_in_layer = flag_size_1_convolution_after_2D_convolution_list[layer_counter];
    flag_batch_normalization_after_size_1_convolution_in_layer = flag_batch_normalization_after_size_1_convolution_list[layer_counter];
    activation_type_str_in_layer = activation_type_str_list[layer_counter];
    
    number_of_kernel_sizes_in_layer = np.size(kernel_size_list[layer_counter]);
    
    conv_model_single_image = vision_block_CONV2D(conv_model_single_image, 
                                             number_of_filters_in_layer, 
                                             kernel_size_in_layer,
                                             flag_dilated_convolution_in_layer,
                                             flag_resnet_in_layer, 
                                             flag_size_1_convolution_on_shortcut_in_resnet_in_layer, 
                                             flag_batch_normalization_in_layer, 
                                             flag_size_1_convolution_after_2D_convolution_in_layer, 
                                             flag_batch_normalization_after_size_1_convolution_in_layer, 
                                             activation_type_str_in_layer, 
                                             );
#END of layers loop  
                                          
#Decide whether put a dense model on top of convolutional model:
flag_put_dense_on_top_of_convolutional_model = 1;
number_of_neurons_per_layer_convolutional_model = [2]; #at the end we need two outputs: {X,Y}
if flag_put_dense_on_top_of_convolutional_model==1:
    conv_model_single_image = mlp_block(conv_model_single_image,number_of_neurons_per_layer_convolutional_model);

#Decide whether to add an Average Layer on top of convolutional model to effectively have
#the model predict optical flow (Mutually exclusive to dense output above!):
flag_use_average_layer_on_top_of_convolutional_model = 0;
if flag_use_average_layer_on_top_of_convolutional_model == 1:
    conv_model_single_image = Average(conv_model_single_image);
    

#TimeDistributed:
#(1). take Conv2D model and actually make it a "Model" according to the functional API
conv_model_single_image_as_model = Model(inputs=[image_input],outputs=[conv_model_single_image])
#(2). make it time distributed or bidirectional(TimeDistributed)
conv_model_time_distributed = TimeDistributed(conv_model_single_image_as_model)(image_inputs);  
#(3). after TimeDistributed we have a tensor of shape (batch_size,number_of_time_steps,single_model_output)
#     so i need to add a top layer to output something of desired shape:
conv_model_time_distributed = Flatten()(conv_model_time_distributed)
conv_model_time_distributed = Dense(2)(conv_model_time_distributed);
#(3). make the whole thing, after TimeDistributed, a Model according to the functional API:
#K.set_learning_phase(0)
conv_model_time_distributed = Model(inputs=[image_inputs],outputs=[conv_model_time_distributed])
conv_model_time_distributed._uses_learning_phase = True #for learning=True, for testing = False

#Visualize Model:
if flag_plot_model == 1:
    keras.utils.plot_model(conv_model_single_image_as_model)
    keras.utils.vis_utils.plot_model(conv_model_single_image_as_model);
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    SVG(model_to_dot(conv_model_single_image_as_model).create(prog='dot', format='svg'))

#Summarize Model:
conv_model_single_image_as_model.summary();
conv_model_time_distributed.summary();


def clip_shift_layer(predicted_shifts, max_shift=1):
#    predicted_shifts[(predicted_shifted > max_shift)] = max_shift; 
    return K.clip(predicted_shifts,-max_shift,max_shift);

def custom_loss_function(predicted_shifts, true_shifts):
    #if i predict images
    max_shift = max_shift_number_global;
    predicted_x = predicted_shifts[0];
    predicted_y = predicted_shifts[1];
    true_x = true_shifts[0];
    true_y = true_shifts[1];
    difference_clipped = K.clip(K.abs(predicted_shifts-true_shifts),min_value=-max_shift,max_value=max_shift);
    return K.mean(K.square(predicted_shifts - true_shifts), axis=-1)
    




#Simulations Parameters:
#ROI=32
#number_of_time_steps = 4
speckle_size = 5 
epp = 10000 
readout_noise = 140
max_shift = 0.1
constant_decorrelation_fraction = 0
Fs = 5500

#Training Parameters:
flag_single_number_or_optical_flow_prediction = 1
flag_center_each_speckles_independently_or_as_batch = 2
flag_stretch_each_speckles_independently_or_as_batch = 2
std_wanted = 1
flag_stateful_batch = 1
flag_online_or_from_file = 1
flag_shuffle = False
file_name_to_get_training_data_from = 'bla'  
training_directory = 'C:/Users\master\.spyder-py3\KERAS'
model_name = 'deep_speckles_simple_cnn.h5'

#Batch size and number of epochs
batch_size = 32;
batch_size_validation = 200;
number_of_epochs = 10;

#Batch Generator:
#Perhapse for fit_generator i need a yield operation, but perhapse i should also have a regular
#generator function which simply returns a batch as a regular function for more functionality
# Parameters
generator_parameters = {'ROI':ROI,
                        'speckle_size':speckle_size ,
                        'epp':epp,
                        'readout_noise':readout_noise ,
                        'max_shift':max_shift,
                        'number_of_time_steps':number_of_time_steps,
                        'batch_size':batch_size,
                        'flag_single_number_or_optical_flow_prediction':flag_single_number_or_optical_flow_prediction ,
                        'constant_decorrelation_fraction':constant_decorrelation_fraction ,
                        'Fs':Fs,
                        'flag_center_each_speckles_independently_or_as_batch':flag_center_each_speckles_independently_or_as_batch ,
                        'flag_stretch_each_speckles_independently_or_as_batch':flag_stretch_each_speckles_independently_or_as_batch ,
                        'std_wanted':std_wanted,
                        'flag_stateful_batch':flag_stateful_batch ,
                        'flag_online_or_from_file':flag_online_or_from_file,
                        'file_name': file_name_to_get_training_data_from ,
                        'flag_shuffle':flag_shuffle }
generator_parameters_validation = generator_parameters;
generator_parameters_validation['batch_size'] = batch_size_validation;

#Generator objects:
training_generator_object = DataGenerator(**generator_parameters)
validation_generator_object = DataGenerator(**generator_parameters)
#Generator:
number_of_batches = 5; #number of batches to generate (to take care of memory concerns etc')
                       #if i use an iterator batch generator this should be 1 by definition(?):
training_generator = training_generator_object.generate_iterator(number_of_batches);
validation_generator = validation_generator_object.generate_iterator(number_of_batches);

###tensorflow session:
#sess = tf.Session()
#K.set_session(sess)
#init_op = tf.global_variables_initializer()
#sess.run(init_op)



###### COMPILE: ######

#Set GLOBAL VARIABLES which go in to all sorts of possible functions:
max_shift_number_global = 3;

#COMPILE MODEL:
#from keras.optimizers import Adam,SGD,RMSprop,Adagrad,Nadam,TFOptimizer
metrics_list = ['mse'];
conv_model_time_distributed.compile(optimizer='adam', loss=custom_loss_function, metrics=metrics_list);

#Fit Model:
flag_method_of_fitting = 3;

#Callbacks:
filepath_string = 'deep_speckles_weights.{epoch:02d}_{val_loss:.2f}.hdf5';
model_checkpoint_function = ModelCheckpoint(filepath_string, period=1, monitor='val_loss',verbose=0,
                                            save_best_only=False,save_weights_only=False,mode='auto');
custom_callback_function = custom_callback();
callbacks_list = [model_checkpoint_function,custom_callback_function];

#Get constant validation set:
validation_X,validation_y = next(validation_generator);

#Fit:
flag_use_iterator_or_function_to_generate_batches = 2;
numer_of_epochs = 10;
if flag_method_of_fitting == 1:
    #(1). Fit using large data already in the form of numpy arrays
    
    #Generate batches:
    if flag_use_iterator_or_function_to_generate_batches==1:
        [X,y] = next(training_generator);
    elif flag_use_iterator_or_function_to_generate_batches==2:
        [X,y] = training_generator_object.generate_function(number_of_batches);
   
    #Fit:
    history = conv_model_time_distributed.fit(x=X, y=y, batch_size=batch_size, epochs=1, 
                                              callbacks = callbacks_list, 
                                              validation_data = [validation_X,validation_y])
    
    #Plot what's going on:
    figure(1);
    subplot(2,1,1);
    plot(history['loss']);
    ylabel('loss');
    xlabel('epochs');
    subplot(2,1,2);
    plot(sqrt(history.history('mse')));
    ylabel('std');
    xlabel('epochs');
    
elif flag_method_of_fitting == 2:
    #(2). Fit using a Generator function:
    number_of_workers = 1;
#    K.set_learning_phase(0)
    conv_model_time_distributed.fit_generator(generator = training_generator,
                                              steps_per_epoch = 100,
                                              validation_data = validation_generator,
                                              validation_steps = 10,
                                              workers = number_of_workers)
elif flag_method_of_fitting == 3:
    #(3). Fitting using one fit per loop (more controllable):
    1;

    
    
    
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






################################################################################################################################################################################################################                                          
                     






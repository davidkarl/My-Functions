# deep_speckles


######   IMPORT RELEVANT LIBRARIES: ###########
#(1). Main Modules
from __future__ import print_function
import keras
from keras import backend as K
#import cv2    NEED TO USE PIP INSTALL!!!!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import os
import datetime
import time #time.time()
import sys
import nltk
import collections
import re
from csv import reader
import tarfile
from pandas import read_csv
from pandas import Series
import collections
#counter = collections.Counter()
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
import argparse
length = len #use length instead of len.... make sure it doesn't cause problems


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
from keras.layers import Merge, RepeatVector, Reshape, TimeDistributed, Concatenate
##########################################################################################################################################################
#(3). Optimizers 
from keras.optimizers import Adam,SGD,RMSprop,Adagrad,Nadam,TFOptimizer
#(4). Callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, BaseLogger, EarlyStopping, LambdaCallback, CSVLogger
#(5). Generators
from keras.preprocessing.image import ImageDataGenerator
#(6). Regularizers
from keras.regularizers import l2
#(7). Normalization (maybe add weight normalization, layer normalization, SELU)
from keras.layers.normalization import BatchNormalization
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
from keras.backend import concatenate


#########  Numpy Functions: ############
#(1). Arrays & Matrices
from numpy import array, arange, asarray, asmatrix, atleast_1d, atleast_2d, atleast_3d, copy
#(2). Apply operators:
from numpy import apply_along_axis, apply_over_axes, transpose
#(3). Finding indices/elements wanted
from numpy import amax, amin, argmin, argmax, argwhere, argsort, where, who
from numpy import equal,greater_equal, greater, not_equal, less_equal
#(4). Mathematical Operations
from numpy import absolute, add, average, exp, exp2, log, log10, log2, mod, real, imag, sqrt, square
from numpy import floor, angle, conj, unwrap
from numpy import mean, median, average, cumsum, std, diff, clip
#(5). Linspace, Meshgrid:
from numpy import meshgrid, linspace, logspace, roll#, roll_axis
#(6). Shape Related:
from numpy import reshape, resize, shape, newaxis, rot90, flip, fliplr, flipud, expand_dims, left_shift
from numpy import squeeze, moveaxis
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




######### Math/Random Functions/Constants: #########
from math import sin
from math import pi
from math import exp
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
####################################################################################################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################


#############################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################
# DATA GENERATOR WITH 2 OPTIONS: 
# (1). generate on the fly using random phase screen
# (2). load from memory files with relevant stuff
# (!) - remember i might want now or later to load several images and not just a pair so
#       i need to incorporate a variable called number_of_images_per_example to be able to load the proper
#       amount of images per example



#############################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################



################################################################################################################################################################
# Parameters:
# Input Parameters:
N = 256;
ROI = 64;
width = ROI;
height = ROI;
input_size = (ROI*ROI,2)
CC_size = 3;
number_of_images = 2; #2 consequtive frames (maybe for turbulence simply use more images instead of LSTM)
#Network Parameters:
number_of_basic_layers = 4;
number_of_filters = ROI;
kernel_size = (5,5,2);
pool_size = 2;
flag_use_dropout = 0;

number_of_filters_dilated = 32;
kernel_size_dilated = 3;
dilation_rate = 2;
#Training Parameters:
training_directory = 'C:/Users\master\.spyder-py3\KERAS'
batch_size = 32;
number_of_epochs = 10;
model_name = 'deep_speckles_simple_cnn.h5'
################################################################################################################################################################################################################



#####################################################################################################################################################################
#####################################################################################################################################################################
########### BASIC CNN LAYERS FOR A SINGLE IMAGE FEATURE EXTRACTION ##############
def vision_block_Conv2D(input_layer, number_of_filters_vec, kernel_size_vec, \
                 flag_dilated_convolution_vec, \
                 flag_resnet_vec, flag_size_1_convolution_on_shortcut_in_resnet_vec, \
                 flag_batch_normalization_vec, \
                 flag_size_1_convolution_after_2D_convolution_vec, flag_batch_normalization_after_size_1_convolution_vec, \
                 activation_type_str_vec):
    
    number_of_kernel_sizes_in_layer = length(kernel_size_vec);
    
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
        
        if flag_dilated_convolution==1:
            vision_block_current_kernel_size = AtrousConvolution2D(number_of_filters, kernel_size, atrous_rate=dilation_rate,border_mode='same')(input_layer);
        else:
            vision_block_current_kernel_size = Conv2D(number_of_filters, kernel_size,padding='same')(input_layer);
            
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
            vision_block_current_kernel_size = Merge([vision_block_current_kernel_size,input_layer],mode='sum')
        
        
        if kernel_size_counter == 0:
            vision_block = vision_block_current_kernel_size;
        else:
            vision_block = Concatenate([vision_block,vision_block_current_kernel_size]);
    #END OF KERNEL SIZE FOR LOOP
       
    return vision_block
#####################################################################################################################################################################
####################################################################################################################################################################



#####################################################################################################################################################################
#####################################################################################################################################################################
########### BASIC MLP BLOCK ##############
def mlp_block(input_layer, number_of_neurons_per_layer):    
    number_of_layers = len(number_of_neurons_per_layer);
    mlp_block = Dense(number_of_neurons_per_layer)(input_layer);
    for layer_counter in arange(1,number_of_layers-1,1):
        mlp_block = Dense(number_of_neurons_per_layer)(mlp_block);
    
    return mlp_block
#####################################################################################################################################################################
#####################################################################################################################################################################





################################################################################################################
####### Build Convolutional model which acts on each image seperately: NO TimeDistributed!!!! #######
######## Conv2D on each image separately ########

    
#Build Network:
#(1). Image Inputs:
number_of_color_channels = 1;
image_input_layers = list()
for image_counter in arange(0,number_of_images,1):
    image_input_layers.append(Input(shape=(ROI,ROI,number_of_color_channels)));


#(1). Convolutional Model:
number_of_convolutional_model_layers = 2;

number_of_filters_vec = (16,8,8);
kernel_size_vec = (5,3,3);
flag_dilated_convolution_vec = (0,0,0);
flag_resnet_vec = (0,0,0);
flag_size_1_convolution_on_shortcut_in_resnet_vec = (0,0,0);
flag_batch_normalization_vec = (1,1,1);
flag_size_1_convolution_after_2D_convolution_vec = (1,1,1);
flag_batch_normalization_after_size_1_convolution_vec = (1,1,1);
activation_type_str_vec = ('relu','relu','relu'); #Consider anti-rectifier

#from vision_block_Conv2D import *

conv_models_output_list = list()
for image_counter in arange(0,number_of_images,1):
    conv_model_current_image = image_input_layers[image_counter];
    for layer_counter in arange(0,number_of_convolutional_model_layers,1):
        #Get parameters for this session:
        number_of_filters = number_of_filters_vec[layer_counter];
        kernel_size = kernel_size_vec[layer_counter];
        flag_dilated_convolution = flag_dilated_convolution_vec[layer_counter];
        flag_resnet = flag_resnet_vec[layer_counter];
        flag_size_1_convolution_on_shortcut_in_resnet = flag_size_1_convolution_on_shortcut_in_resnet_vec[layer_counter];
        flag_batch_normalization = flag_batch_normalization_vec[layer_counter];
        flag_size_1_convolution_after_2D_convolution = flag_size_1_convolution_after_2D_convolution_vec[layer_counter];
        flag_batch_normalization_after_size_1_convolution = flag_batch_normalization_after_size_1_convolution_vec[layer_counter];
        activation_type_str = activation_type_str_vec[layer_counter];
        
        conv_model_current_image = vision_block_Conv2D(conv_model_current_image, 
                                                         number_of_filters, 
                                                         kernel_size,
                                                         flag_dilated_convolution, 
                                                         flag_resnet, 
                                                         flag_size_1_convolution_on_shortcut_in_resnet, 
                                                         flag_batch_normalization, 
                                                         flag_size_1_convolution_after_2D_convolution, 
                                                         flag_batch_normalization_after_size_1_convolution, 
                                                         activation_type_str, 
                                                         );
         #END of layers loop         
    conv_models_output_list.append(Model(inputs=image_input_layers[image_counter] , outputs=[conv_model_current_image]));
    #NOW EACH conv_models[i] is the conv model being applied to a single image

#Concetenate model results of each image to a single tensor output:
conv_model_Conv2D_outputs_concatenated = conv_models_output_list[0];
for i in arange(1,length(conv_models_output_list),1):
    conv_model_Conv2D_outputs_concatenated = concatenate([conv_model_Conv2D_outputs_concatenated,conv_models_output_list[i]]);

#for i in arange(1,number_of_images,1):
#    input_tensor

conv_model_Conv2D_outputs_concatenated = Model(inputs=[image_input_layers], \
                                          outputs=[conv_model_Conv2D_outputs_concatenated]);
################################################################################################################################################################################################################
















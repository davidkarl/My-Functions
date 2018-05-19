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



################################################################################################################################################################################################################
#Different model architectures
#1. vision model for each image to dense concetanated with CC mlp
#2. vision model for each image to CC layer to global dense concetanated with CC mlp
#3. combined 3D convolutional vision model (perhapse more then 2 images) with dense layer + CC mlp
#4. vision model for each image to CC layer same mlp to each CC to dense conct with CC mlp
#5. vision model per image with auxiliary per pixel prediction loss and dense conct with CC mlp
#6. combined 3D conv model with per pixel prediction 
#7. model with per pixel prediction with a LOCAL MLP(?) with CC mlp as conct input

#Different ideas for convolutional vision model:
#1. using dilated convolution in vision model
#2. using fft inputs
#3. using resnets
#4. using different activations (anti-rect, leaky-relu)
#5. using batch normalization/layer normalization/weight normalization

#Different MLP model for correlation:
#1. using D+ cross correlation instead of NCC
################################################################################################################################################################################################################




################################################################################################################################################################################################################

#####################################################################################################################################################################
#####################################################################################################################################################################
########### BASIC CNN LAYERS FOR A SINGLE IMAGE FEATURE EXTRACTION ##############
def vision_block_CONV2D(input_layer, number_of_filters_vec, kernel_size_vec, \
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
#####################################################################################################################################################################



#####################################################################################################################################################################
#####################################################################################################################################################################
########### ConvLSTM2D Layer  ##############
    
#PROBABLY I SHOULD HAVE BOTH A FORWARD AND A BACKWARD FLAG JUST
#TO KEEP THE FLEXIBILITY EVEN THOUGH IT'S USUALLY EITHER ONLY FORWARD OR FORWARD+BACKWARD (BIDIRECTIONAL)
def vision_block_ConvLSTM2D(input_layer, number_of_filters_vec, kernel_size_vec, \
                 flag_resnet_vec, flag_size_1_convolution_on_shortcut_in_resnet_vec, \
                 flag_batch_normalization_vec, \
                 flag_size_1_convolution_after_2D_convolution_vec, flag_batch_normalization_after_size_1_convolution_vec, \
                 activation_type_str_vec, \
                 flag_return_sequences_vec, \
                 flag_bidirectional_vec, \
                 flag_stateful_vec):
    
    
    number_of_kernel_sizes_in_layer = length(kernel_size_vec);
    
    
    for kernel_size_counter in arange(0,number_of_kernel_sizes_in_layer,1):
        #Get parameters for each kernel size filters in layer:
        number_of_filters = number_of_filters_vec[kernel_size_counter];
        kernel_size = kernel_size_vec[kernel_size_counter]; 
        flag_resnet = flag_resnet_vec[kernel_size_counter];
        flag_size_1_convolution_on_shortcut_in_resnet = flag_size_1_convolution_on_shortcut_in_resnet_vec[kernel_size_counter];
        flag_batch_normalization = flag_batch_normalization_vec[kernel_size_counter];
        flag_size_1_convolution_after_2D_convolution = flag_size_1_convolution_after_2D_convolution_vec[kernel_size_counter];
        flag_batch_normalization_after_size_1_convolution = flag_batch_normalization_after_size_1_convolution_vec[kernel_size_counter];
        activation_type_str = activation_type_str_vec[kernel_size_counter];
        flag_return_sequences = flag_return_sequences_vec[kernel_size_counter];
        flag_bidirectional = flag_bidirectional_vec[kernel_size_counter];
        flag_stateful = flag_stateful_vec[kernel_size_counter];
            
        if flag_bidirectional == 1:
            number_of_time_directions = 2;
        else:
            number_of_time_directions = 1;
        
        for time_direction_counter in arange(0,number_of_time_directions,1):
            
            vision_block_current_kernel_size = ConvLSTM2D(number_of_filters, kernel_size,padding='same',return_sequences=flag_return_sequences,stateful=flag_stateful,go_backwards=time_direction_counter)(input_layer);
                
            if flag_batch_normalization==1:
                vision_block_current_kernel_size = BatchNormalization()(vision_block_current_kernel_size);
            
            vision_block_current_kernel_size = Activation(activation_type_str)(vision_block_current_kernel_size);
                
            if flag_size_1_convolution_after_2D_convolution==1:
                vision_block_current_kernel_size = ConvLSTM2D(number_of_filters, 1, padding='same',return_sequences=flag_return_sequences,stateful=flag_stateful,go_backwards=time_direction_counter)(vision_block_current_kernel_size);
                if flag_batch_normalization_after_size_1_convolution==1:
                    vision_block_current_kernel_size = BatchNormalization()(vision_block_current_kernel_size);
                vision_block_current_kernel_size = Activation(activation_type_str)(vision_block_current_kernel_size);
                
            
            if flag_resnet==1:
                if flag_size_1_convolution_on_shortcut_in_resnet==1:
                    input_layer = ConvLSTM2D(number_of_filters,1,border_mode='same',return_sequences=flag_return_sequences,stateful=flag_stateful,go_backwards=time_direction_counter);    
                    if flag_batch_normalization_after_size_1_convolution==1:
                        input_layer = BatchNormalization()(input_layer);
                vision_block_current_kernel_size = Merge([vision_block_current_kernel_size,input_layer],mode='sum')
            
            
            if kernel_size_counter == 0:
                vision_block = vision_block_current_kernel_size;
            else:
                vision_block = Concatenate([vision_block,vision_block_current_kernel_size]);
        #END OF TIME DIRECTION FOR LOOP
    #END OF KERNEL SIZE FOR LOOP
       
    return vision_block
#####################################################################################################################################################################
#####################################################################################################################################################################



#####################################################################################################################################################################
#####################################################################################################################################################################
########### Conv3D Layer  ##############
def vision_block_Conv3D(input_layer, number_of_filters_vec, kernel_size_vec_list, \
                         flag_resnet_vec, flag_size_1_convolution_on_shortcut_in_resnet_vec, \
                         flag_batch_normalization_vec, \
                         flag_size_1_convolution_after_2D_convolution_vec, flag_batch_normalization_after_size_1_convolution_vec, \
                         activation_type_str_vec):
    
    #as it is now the kernel_size_vec is a vec intended to have each element be the width and height
    #of the kernel, but when it comes to 3D i should have each element of the kernel_size_vec
    #be a 3-element-tupe of kernel_size_vec[i]=(height,width,depth).
    #so i will change the kernel_size_vec to kernel_size_vec_list.
    
    #i only allow each element of kernel_size_vec_list to be length 1 of length 3
    
    #the kernel size in kernel_size_vec should be a 3-element-tuple or i can use the depth of the
    #input layer and have that be the kernel depth
    
    number_of_kernel_sizes_in_layer = length(kernel_size_vec_list);
    layer_shape = input_layer.output_shape();
    layer_depth = layer_shape[3];
    
    for kernel_size_counter in arange(0,number_of_kernel_sizes_in_layer,1):
        #Get parameters for each kernel size filters in layer:
        kernel_size_vec = kernel_size_vec_list[kernel_size_counter]; 
        if length(kernel_size_vec)!=3:
            if length(kernel_size_vec)==1:
                kernel_size = (kernel_size_vec,kernel_size_vec,layer_depth);
            elif length(kernel_size_vec)==3:
                kernel_size = kernel_size_vec;
        
        number_of_filters = number_of_filters_vec[kernel_size_counter];
        flag_resnet = flag_resnet_vec[kernel_size_counter];
        flag_size_1_convolution_on_shortcut_in_resnet = flag_size_1_convolution_on_shortcut_in_resnet_vec[kernel_size_counter];
        flag_batch_normalization = flag_batch_normalization_vec[kernel_size_counter];
        flag_size_1_convolution_after_2D_convolution = flag_size_1_convolution_after_2D_convolution_vec[kernel_size_counter];
        flag_batch_normalization_after_size_1_convolution = flag_batch_normalization_after_size_1_convolution_vec[kernel_size_counter];
        activation_type_str = activation_type_str_vec[kernel_size_counter];
        

        vision_block_current_kernel_size = Conv3D(number_of_filters, kernel_size,padding='same')(input_layer);
            
        if flag_batch_normalization==1:
            vision_block_current_kernel_size = BatchNormalization()(vision_block_current_kernel_size);
        
        vision_block_current_kernel_size = Activation(activation_type_str)(vision_block_current_kernel_size);
            
        if flag_size_1_convolution_after_2D_convolution==1:
            vision_block_current_kernel_size = Conv3D(number_of_filters, 1, padding='same')(vision_block_current_kernel_size);
            if flag_batch_normalization_after_size_1_convolution==1:
                vision_block_current_kernel_size = BatchNormalization()(vision_block_current_kernel_size);
            vision_block_current_kernel_size = Activation(activation_type_str)(vision_block_current_kernel_size);
            
        
        if flag_resnet==1:
            if flag_size_1_convolution_on_shortcut_in_resnet==1:
                input_layer = Conv3D(number_of_filters,1,border_mode='same');    
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
#####################################################################################################################################################################




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





#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################    





################################################################################################################
####### Build Convolutional model which acts on each image seperately: NO TimeDistributed!!!! #######

######## Conv2D on each image separately ########

#### ADD POSSIBILITY TO HAVE A FEW CONCATENATED FILTER SIZE OUTPUTS TO EACH LAYERS: ####
    
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
        
        conv_model_current_image = vision_block_CONV2D(conv_model_current_image, 
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
conv_model_Conv2D_outputs_concatenated = Model(inputs=[image_input_layers[0],image_input_layers[1]], \
                                          outputs=[conv_model_Conv2D_outputs_concatenated]);
################################################################################################################################################################################################################

                                          
                                          
                                          
################################################################################################################################################################################################################
#### USING TIME DISTRIBUTED LAYER - DOING IT RIGHT UNLIKE WHAT I DID ABOVE: ######

                                        
#Build Network:
#(1). Image Input:
number_of_color_channels = 1;
#image_input = Input(shape=(ROI,ROI,number_of_color_channels));
image_input = Input(shape=(number_of_images,ROI,ROI,number_of_color_channels)); #????????


#Convolutional Model:
number_of_convolutional_model_layers = 3;


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
#Layer 1 (i use several sized filters to analyze the speckles on several typical sizes and
#         maybe differentiate different speckle sizes or differentiate speckles from background noise)
kernel_size_list.append((7,5,3,1));
number_of_filters_list.append((4,4,4,4));
flag_dilated_convolution_list.append((0,0,0,0));
flag_resnet_list.append((0,0,0,0));
flag_batch_normalization_list.append((1,1,1,1));
flag_size_1_convolution_on_shortcut_in_resnet_list.append((0,0,0,0));
flag_size_1_convolution_after_2D_convolution_list.append((0,0,0,0));
flag_batch_normalization_after_size_1_convolution_list.append((1,1,1,1));
activation_type_str_list.append(('relu','relu','relu','relu')); 
#Layer 2:
kernel_size_list.append((3));
number_of_filters_list.append((5));
flag_dilated_convolution_list.append((0));
flag_resnet_list.append((0));
flag_batch_normalization_list.append((1));
flag_size_1_convolution_on_shortcut_in_resnet_list.append((0));
flag_size_1_convolution_after_2D_convolution_list.append((0));
flag_batch_normalization_after_size_1_convolution_list.append((1));
activation_type_str_list.append(('relu'));  
#Layer 3 (one output image/filter, useful for fully-convolutional networks):
kernel_size_list.append((3));
number_of_filters_list.append((1));
flag_dilated_convolution_list.append((0));
flag_resnet_list.append((0));
flag_batch_normalization_list.append((1));
flag_size_1_convolution_on_shortcut_in_resnet_list.append((0));
flag_size_1_convolution_after_2D_convolution_list.append((0));
flag_batch_normalization_after_size_1_convolution_list.append((1));
activation_type_str_list.append(('relu'));  


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
    
    number_of_kernel_sizes_in_layer = length(kernel_size_list[layer_counter]);
    
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
#IF I PUT A DENSE MODEL ON TOP OF THE CONV2D WILL HOW WILL IT INTERFER WITH THE TIMEDISTRIBUTED????
#SHOULD I PUT A TimeDistributed(layer) in each layer?!!!?!?!
flag_put_dense_on_top_of_convolutional_model = 0;
number_of_neurons_per_layer_convolutional_model = (9);
if flag_put_dense_on_top_of_convolutional_model==1:
    conv_model_single_image = mlp_block(conv_model_single_image,number_of_neurons_per_layer_convolutional_model);

#Decide whether to add an Average Layer on top of convolutional model to effectively have
#the model predict optical flow (Mutually exclusive to dense output above!):
flag_use_average_layer_on_top_of_convolutional_model = 0;
if flag_use_average_layer_on_top_of_convolutional_model == 1:
    conv_model_single_image = Average(conv_model_single_image);
    

#TimeDistributed:
flag_bidirectional = 1;
if flag_bidirectional == 1:
    conv_model_time_distributed = Bidirectional(TimeDistributed(conv_model_single_image));
else:
    conv_model_time_distriubted = TimeDistributed(conv_model_single_image);     
################################################################################################################################################################################################################                                          
                         



################################################################################################################################################################################################################
#### USING ConvLSTM2D ######

                                        
#Build Network:
#(1). Image Input:
number_of_color_channels = 1;
#image_input = Input(shape=(ROI,ROI,number_of_color_channels));
image_input = Input(shape=(number_of_images,ROI,ROI,number_of_color_channels));


#Convolutional Model:
number_of_convolutional_model_layers = 3;


#Get Layer Parameters by specifying each layer parts separately (less compact but probably better):
#
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
flag_return_sequences_list = list();
flag_bidirectional_list = list();
flag_stateful_list = list();
#Layer 1 (i use several sized filters to analyze the speckles on several typical sizes and
#         maybe differentiate different speckle sizes or differentiate speckles from background noise)
kernel_size_list.append((7,5,3,1));
number_of_filters_list.append((4,4,4,4));
flag_dilated_convolution_list.append((0,0,0,0));
flag_resnet_list.append((0,0,0,0));
flag_batch_normalization_list.append((1,1,1,1));
flag_size_1_convolution_on_shortcut_in_resnet_list.append((0,0,0,0));
flag_size_1_convolution_after_2D_convolution_list.append((0,0,0,0));
flag_batch_normalization_after_size_1_convolution_list.append((1,1,1,1));
activation_type_str_list.append(('relu','relu','relu','relu')); 
flag_return_sequences_list.append((true,true,true,true));
flag_bidirectional_list.append((true,true,true,true));
flag_stateful_list.append(true,true,true,true); #Remember to flush hidden states manually if i want
#Layer 2:
kernel_size_list.append((3));
number_of_filters_list.append((5));
flag_dilated_convolution_list.append((0));
flag_resnet_list.append((0));
flag_batch_normalization_list.append((1));
flag_size_1_convolution_on_shortcut_in_resnet_list.append((0));
flag_size_1_convolution_after_2D_convolution_list.append((0));
flag_batch_normalization_after_size_1_convolution_list.append((1));
activation_type_str_list.append(('relu'));  
flag_return_sequences_list.append((true));
flag_bidirectional_list.append((true));
flag_stateful_list.append(true); #Remember to flush hidden states manually if i want
#Layer 3 (one output image/filter, useful for fully-convolutional networks):
kernel_size_list.append((3));
number_of_filters_list.append((1));
flag_dilated_convolution_list.append((0));
flag_resnet_list.append((0));
flag_batch_normalization_list.append((1));
flag_size_1_convolution_on_shortcut_in_resnet_list.append((0));
flag_size_1_convolution_after_2D_convolution_list.append((0));
flag_batch_normalization_after_size_1_convolution_list.append((1));
activation_type_str_list.append(('relu'));  
flag_return_sequences_list.append((false));
flag_bidirectional_list.append((true));
flag_stateful_list.append(true); #Remember to flush hidden states manually if i want


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
    flag_return_sequences = flag_return_sequences_list[layer_counter];
    flag_go_backwards = flag_go_backwards_list[layer_counter];
    flag_stateful = flag_go_backwards_list[layer_counter];
    
    number_of_kernel_sizes_in_layer = length(kernel_size_list[layer_counter]);
    
    conv_model_single_image = vision_block_ConvLSTM2D()
                                             conv_model_single_image, 
                                             number_of_filters_in_layer, 
                                             kernel_size_in_layer,
                                             flag_resnet_in_layer, 
                                             flag_size_1_convolution_on_shortcut_in_resnet_in_layer, 
                                             flag_batch_normalization_in_layer, 
                                             flag_size_1_convolution_after_2D_convolution_in_layer, 
                                             flag_batch_normalization_after_size_1_convolution_in_layer, 
                                             activation_type_str_in_layer, 
                                             flag_return_sequences, \
                                             flag_go_backwards, \
                                             flag_stateful
                                             );
#END of layers loop  
                                          
#Decide whether put a dense model on top of convolutional model:
#IF I PUT A DENSE MODEL ON TOP OF THE CONV2D WILL HOW WILL IT INTERFER WITH THE TIMEDISTRIBUTED????
#SHOULD I PUT A TimeDistributed(layer) in each layer?!!!?!?!
flag_put_dense_on_top_of_convolutional_model = 0;
number_of_neurons_per_layer_convolutional_model = (9);
if flag_put_dense_on_top_of_convolutional_model==1:
    conv_model_single_image = mlp_block(conv_model_single_image,number_of_neurons_per_layer_convolutional_model);

#Decide whether to add an Average Layer on top of convolutional model to effectively have
#the model predict optical flow (Mutually exclusive to dense output above!):
flag_use_average_layer_on_top_of_convolutional_model = 0;
if flag_use_average_layer_on_top_of_convolutional_model == 1:
    conv_model_single_image = Average(conv_model_single_image);
    

#TimeDistributed:
conv_model_time_distriubted = TimeDistributed(conv_model_single_image);     
################################################################################################################################################################################################################                                          
                         


                 

################################################################################################################################################################################################################
##### BUILDING THE MLP FOR THE CROSS CORRELATION: #####
number_of_mlp_model_layers = 4;
number_of_neurons_per_layer = (CC_size**2,CC_size**2,CC_size**2,CC_size**2)
flag_use_mlp_model_or_just_use_CC_itself = 1;
flag_concatenate_CC_input_to_output = 1;
#(1). Cross Correlation Inputs:
cross_correlation_input = Input(shape=(CC_size,CC_size))
cross_correlation_output_single_image_pair = mlp_block(cross_correlation_input,number_of_neurons_per_layer);
#TimeDistriubted - Notice that the number of outputs will be (number_of_images-1) if 
#                  done for consecutive pairs (could possibly be done for each image pairs which is n!):
cross_correlation_output_time_distributed = TimeDistributed(cross_correlation_output_single_image_pair);
################################################################################################################################################################################################################





#vision_block = Model(inputs=[image_input_layers[0],image_input_layers[1]], outputs=vision_block);






########################################################
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
########################################################



#define Network:
#Start creating the KERAS network using Sequantial:
model = Sequential()
model.add(Conv3D(number_of_filters, kernel_size, padding='same', #32 is the number of filters!!!
                 input_shape = x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512)) #512 number of output neurons 
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
current_optimizer = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=current_optimizer,
              metrics=['accuracy'])























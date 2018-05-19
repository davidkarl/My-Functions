
##
#<editor-fold desc="Description">
# import importlib
# import sys
# for module in sys.modules.values():
#     importlib.reload(module)

# #solution1:
# import sys
# sys.modules.clear()

# solution2:
# import sys
# if globals().has_key('init_modules'):
# 	for m in [x for x in sys.modules.keys() if x not in init_modules]:
# 		del(sys.modules[m])
# else:
# 	init_modules = sys.modules.keys()

# import clear_all
# exec(clear_all.clear_all)

##########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
######   IMPORT RELEVANT LIBRARIES: ###########
#(1). Main Modules
# from __future__ import print_function
import keras
from keras import backend as K
import tensorflow as tf
#import cv2    NEED TO USE PIP INSTALL!!!!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
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
import struct
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
#(14). Applications:
# from keras.applications.xception import Xception
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.densenet import DenseNet121
# from keras.applications.densenet import DenseNet169
# from keras.applications.densenet import DenseNet201
# from keras.applications.nasnet import NASNetLarge


#(15). Metrics: mse, mae, mape, cosine
#Mean Squared Error: mean_squared_error, MSE or mse
#Mean Absolute Error: mean_absolute_error, MAE, mae
#Mean Absolute Percentage Error: mean_absolute_percentage_error, MAPE, mape
#Cosine Proximity: cosine_proximity, cosine

##Example of Usage:
#model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])
#history = model.fit(X, X, epochs=500, batch_size_file=len(X), verbose=2)
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
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################
##
#</editor-fold>


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
import get_speckle_sequences_full
from get_speckle_sequences_full import *
import modify_speckle_sequences
from modify_speckle_sequences import *
import search_file
from search_file import *
import tic_toc
from tic_toc import *
import klepto_functions
from klepto_functions import *
import clear_all
from clear_all import clear_all
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################

#i will use a "neutral" simulation which can be modified later in the main training scripts:
#epp=1, no noises!

##### Decide whether to add to presaved .bin file or write new file:
######
flag_continue_adding_to_previous_bin_file_or_generate_new = 2; #1=continue from previous, 2=new
speckles_file_name_to_continue_adding_to = 'speckle_matrices_1'; #get name you want from work folder
######

########################################################################################################
#get variables as dictionary and then as lists of keys and values
_baseline_variables = locals().copy();
_baseline_variables_keys = list(_baseline_variables.keys());
_baseline_variables_values = list(_baseline_variables.values());
_baseline_variables_keys_set = set(_baseline_variables_keys);
########################################################################################################


#Speckle parameters:
data_type = 'f';
ROI=32
speckle_size = 5
number_of_time_steps = 4

#Sizes:
batch_size_file = 32
number_of_different_batches = 100; #if flag_stateful_batch = 0 then i could just as well have batch_size_file=1
                                    #and have the number of dfferent batches be the number of samples
number_of_samples = batch_size_file * number_of_different_batches;

#Simulation parameters:
max_shift = 0.1
constant_decorrelation_fraction = 0
Fs = 5500
flag_stateful_batch = 0
stateful_batch_size = batch_size_file;

#Label/Output parameters:
flag_single_number_or_optical_flow_prediction = 1; #1=single number, 2=optical flow

#Noises:
epp = 1;
readout_noise = 0
background = 0;
flag_use_shot_noise = 0;
cross_talk = 0;

#Modification parameters:
flag_center_each_speckles_independently_or_as_batch_example_from_file = 3 # 3= do nothing!!!!!
flag_stretch_each_speckles_independently_or_as_batch_example_from_file = 3 # 3= do nothing!!!!!
std_wanted_from_file = 1


########################################################################################################
#Save current session variables:
_final_variables = locals().copy();
_final_variables_keys = list(_final_variables.keys());
_final_variables_values = list(_final_variables.values());
_final_variables_keys_set = set(_final_variables_keys);
#get dictionary with relevant variable names and their values:
final_variables_dictionary = dict();
for k,v in zip(_final_variables_keys,_final_variables_values):
        final_variables_dictionary[k] = v;
#get only relevant keys by removing union:
relevant_keys_set = _final_variables_keys_set ^ _baseline_variables_keys_set;
relevant_keys_set = [element for element in relevant_keys_set if element.startswith('_')==False]
relevant_keys_set = list(relevant_keys_set);
relevant_variables_dictionary = {key:final_variables_dictionary[key] for key in relevant_keys_set}
########################################################################################################



#ACT on whether to continue adding matrices or start new file:
if flag_continue_adding_to_previous_bin_file_or_generate_new == 1:
    #Continue previouse .bin file:

    X_file_version = speckles_file_name_to_continue_adding_to[-1];
    file_name_X = speckles_file_name_to_continue_adding_to;
    file_name_y = 'speckle_shifts_' + X_file_version;
    #use that version's klepto file to load all necessary variables:
    load_variables_from_klepto_file(speckles_file_name_to_continue_adding_to + '_klepto.txt');

elif flag_continue_adding_to_previous_bin_file_or_generate_new == 2:
    #Create new .bin file:

    #(1). Find out how many speckle matrices .bin files there are:
    name_start_to_look_for = 'speckle_matrices';
    file_names_list_of_speckle_matrices = search_file(name_start_to_look_for + '*.bin');
    number_of_speckle_matrices_files_so_far = length(file_names_list_of_speckle_matrices);
    #(2). Save variables to appropriate klepto file:
    matrices_file_name_start = 'speckle_matrices_' + str(number_of_speckle_matrices_files_so_far);
    shifts_file_name_start = 'speckle_shifts_' + str(number_of_speckle_matrices_files_so_far);
    klepto_file_name = matrices_file_name_start + '_klepto';
    save_variables_to_klepto_file(klepto_file_name, relevant_variables_dictionary)
    #(3). Speckles (X) and shifts (y) file names:
    file_name_X = matrices_file_name_start + '.bin';
    file_name_y = shifts_file_name_start + '.bin';


#Previous version stuff:
##
#File name and binary file stuff:
# flag1 = flag_single_number_or_optical_flow_prediction
# flag2 = flag_center_each_speckles_independently_or_as_batch_example_from_file
# flag3 = flag_stretch_each_speckles_independently_or_as_batch_example_from_file

# file_name_X = 'speckle_matrices' + \
#             '$' + str(ROI) + \
#             '$' + str(speckle_size) + \
#             '$' + str(epp) + \
#             '$' + str(readout_noise) + \
#             '$' + str(background) + \
#             '$' + str(number_of_time_steps) + \
#             '$' + str(flag_single_number_or_optical_flow_prediction) + \
#             '$' + str(constant_decorrelation_fraction) + \
#             '$' + str(Fs) + \
#             '$' + str(flag_center_each_speckles_independently_or_as_batch_example_from_file) + \
#             '$' + str(flag_stretch_each_speckles_independently_or_as_batch_example_from_file) + \
#             '$' + str(std_wanted_from_file) + \
#             '$' + str(flag_stateful_batch) + \
#             '$' + str(batch_size_file) + \
#             '$' + str(data_type) + '.bin';
# file_name_y = 'speckle_shifts' + file_name_X[16:];

# dictionary_from_filename_to_variables = {'ROI',
#                                          'speckle_size',
#                                          'epp',
#                                          'readout_noise',
#                                          'background',
#                                          'readout_noise',
#                                          'number_of_time_steps',
#                                          'flag_single_number_or_optical_flow_prediction',
#                                          'constant_decorrelation_fraction',
#                                          'Fs',
#                                          'flag_center_each_speckles_independently_or_as_batch_example_from_file',
#                                          'flag_stretch_each_speckles_independently_or_as_batch_example_from_file',
#                                          'std_wanted_from_file',
#                                          'flag_stateful_batch',
#                                          'batch_size_file'
#                                          'data_type'
#                                         }
##

#file_name = 'speckle_matrices'+'ROI='+str(ROI);



if os.path.exists(os.getcwd()+'/'+file_name_X) == 1:
    fid_write_X = open(file_name_X,'ab');
    fid_write_y = open(file_name_y,'ab');
else:
    fid_write_X = open(file_name_X,'wb'); #wb=overwrite, a=append
    fid_write_y = open(file_name_y,'wb'); #wb=overwrite, a=append

#Create dataset:
for batch_counter in arange(0,number_of_different_batches):



    X,y = get_speckle_sequences_full(ROI,
                                     speckle_size,
                                     max_shift,
                                     number_of_time_steps,
                                     batch_size_file, \
                                     flag_single_number_or_optical_flow_prediction, \
                                     constant_decorrelation_fraction,
                                     Fs,\
                                     flag_stateful_batch, \
                                     data_type);


    X = modify_speckle_sequences(X,\
                                 epp,\
                                 readout_noise,\
                                 background,\
                                 flag_use_shot_noise,\
                                 cross_talk,\
                                 flag_center_each_speckles_independently_or_as_batch_example_from_file,\
                                 flag_stretch_each_speckles_independently_or_as_batch_example_from_file,\
                                 std_wanted_from_file,\
                                 flag_stateful_batch,\
                                 stateful_batch_size);


#    #Doesn't save the way i want it:
#    X.tofile(fid_write_X);
#    y.tofile(fid_write_y);

    #(1). write down images:
    #maybe instead of loops i can do this in a more effecient way which will give me the same structure?
    #something like dtype=(float32*32,float32*4,.......)
    for batch_sample_counter in arange(0,batch_size_file):
        for time_counter in arange(0,number_of_time_steps):
#            s1 = struct.pack(data_type*(ROI*ROI),*np.ndarray.flatten(X[batch_sample_counter,time_counter,:,:,0]));
#            fid_write.write(s1);
            X[batch_sample_counter,time_counter,:,:,0].tofile(fid_write_X);


#    #(2). write down shift in middle image:
    for batch_sample_counter in arange(0,batch_size_file):
        y[batch_sample_counter,:].tofile(fid_write_y);
###            s2 = struct.pack(data_type,*np.ndarray.flatten(np.array(y[batch_sample_counter,0])));
###            s2 = struct.pack(data_type,*np.ndarray.flatten(np.array(y[batch_sample_counter,1])));
###            fid_write.write(s2);



#Close file:
fid_write_X.close();
fid_write_y.close();



##Read file:
#number_of_batches_to_read = 1;
#number_of_batch_samples = number_of_batches_to_read * batch_size_file;
#number_of_images_to_read = number_of_batch_samples * number_of_time_steps;
#ROI_shape = (ROI,ROI);
#mat_shape = np.append(number_of_batch_samples,number_of_time_steps);
#mat_shape = np.append(mat_shape,ROI_shape)
#mat_shape = np.append(mat_shape,1)
#single_image_number_of_elements = np.prod(ROI_shape);
#total_images_number_of_elements = single_image_number_of_elements*number_of_images_to_read;
#
#if data_type == 'd':
#    bytes_per_element = 8; #double
#elif data_type == 'f':
#    bytes_per_element = 4; #float
#
#
#fid_read_X = open(file_name_X,'rb');
#fid_read_y = open(file_name_y,'rb');




#SLOW!!!!
#tic()
#images_read_as_bytes = fid_read_X.read(total_images_number_of_elements*bytes_per_element)
#unpack_length = int(length(images_read_as_bytes)/bytes_per_element);
#images_float_tuple_flattened = struct.unpack(data_type*unpack_length, images_read_as_bytes) #returns a tuple...not an array :(
#images_float_array_flattened = np.array(images_float_tuple_flattened,dtype=data_type);
#images_float_array = images_float_array_flattened.reshape(mat_shape,order='C')
#toc()



###read using numpy.fromfile and TESTING:
#tic()
## Parameters:
#batch_size_to_read_temp = 1;
#number_of_batches_to_read = 1;
#total_number_of_samples = number_of_batches_to_read * batch_size_to_read_temp;
#number_of_images_to_read_including_time_steps = total_number_of_samples * number_of_time_steps;
#ROI_shape = (ROI,ROI);
#single_image_number_of_elements = np.prod(ROI_shape);
#total_images_number_of_elements = single_image_number_of_elements*number_of_images_to_read_including_time_steps;
#mat_shape = np.append(total_number_of_samples,number_of_time_steps);
#mat_shape = np.append(mat_shape,ROI_shape)
#mat_shape = np.append(mat_shape,1)
## Read:
#fid_read_X = open(file_name_X,'rb');
#fid_read_y = open(file_name_y,'rb');
#bla_np = np.fromfile(fid_read_X,'f',count=total_images_number_of_elements);
#bla_np = bla_np.reshape(mat_shape);
#y_np = np.fromfile(fid_read_y,'f',count=batch_size_to_read_temp*2)
#y_np = y_np.reshape((batch_size_to_read_temp,2))
###HOW CAN I DEFINE PREDEFINED "SLICES" WHICH COULD ALSO INCLUDE "THE ENTIRE RANGE IN THIS AXIS ALA :"
#figure(1)
#imshow(bla_np[2,2,:,:,0]);
#title(str( y_np[5]) + ' vs. ' + str(y[5]) )
#figure(2)
#imshow(X[2,2,:,:,0]);
#toc()
## Close fid
#fid_read_X.close()
#fid_read_y.close()









#imshow(bla_np.reshape(ROI_shape));
##using array.array
#bla_double_array = array.array('d',bla);











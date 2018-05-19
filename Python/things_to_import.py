# def things_to_import():


######   IMPORT RELEVANT LIBRARIES: ###########
#(1). Main Modules
# from __future__ import print_function
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


#if __name__ == "__main__":
#    # execute only if run as a script
#    length = len #use length instead of len.... make sure it doesn't cause problems
#    import int_range, int_arange, mat_range, matlab_arange, my_linspace, my_linspace_int,importlib
#    from int_range import *
#    from int_arange import *
#    from mat_range import * 
#    from matlab_arange import *
#    from my_linspace import *
#    from my_linspace_int import *
#    from importlib import reload
#    end = -1;
#    start = -1;
#    import get_center_number_of_pixels
#    from get_center_number_of_pixels import *



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
from numpy import absolute, power, add, average, exp2, log, log10, log2, mod, real, imag, sqrt, square
from numpy import floor, angle, conj, unwrap
from numpy import mean, median, average, cumsum, std, diff, clip
#(5). Linspace, Meshgrid:
from numpy import meshgrid, linspace, logspace, roll#, roll_axis
#(6). Shape Related:
from numpy import reshape, resize, shape, newaxis, rot90, flip, fliplr, flipud, expand_dims, left_shift
from numpy import squeeze, moveaxis #, flatten
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
from pylab import imshow, pause, draw, title, axes, ylabel, ylim, yticks, xlabel, xlim, xticks
from pylab import colorbar, colormaps, colors, subplot, suptitle, legend


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
length = len #use length instead of len.... make sure it doesn't cause problems
import int_range, int_arange, mat_range, matlab_arange, my_linspace, my_linspace_int,importlib
from int_range import *
from int_arange import *
from mat_range import * 
from matlab_arange import *
from my_linspace import *
from my_linspace_int import *
end = -1;
start = -1;
#import get_center_number_of_pixels
#from get_center_number_of_pixels import *
#import get_speckle_sequences
#from get_speckle_sequences import *
#import show_matrices_video
#from show_matrices_video import *
#import get_center_number_of_pixels
#from get_center_number_of_pixels import *
####################################################################################################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################




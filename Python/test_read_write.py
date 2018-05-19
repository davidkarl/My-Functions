

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
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################

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
######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################



#Create an array:
X,y = get_speckle_sequences();


######################################################################################################################################################################################################
#Using binary (fwrite fread):
#(1). WRITE:
#(a). fid.write: (it doesn't write the way i want....)
file_name = 'speckle_matrices_3.bin';
fid_write = open(file_name,'wb'); #wb=overwrite, a=append
for batch_counter in arange(0,32,1):    
    for time_counter in arange(0,4,1):
        fid_write.write(X[batch_counter,time_counter,:,:,0]); #THIS ISN'T BEING SAVED THE WAY I WANT
fid_write.close();
#(b). np.ndarray.tofile
#X.tofile(fid_write)
#(c). HD5F:




#(2). READ:
import array
#read using fid.read and struct.unpack:
bytes_per_element = struct.calcsize('d');
number_of_images_to_read = 3;
ROI_shape = (64,64);
mat_shape = np.append(number_of_images_to_read,ROI_shape)
read_size = np.prod(ROI_shape)*number_of_images_to_read;
fid_read = open(file_name,'rb');
bla = fid_read.read(read_size*bytes_per_element)
unpack_length = int(length(bla)/8);
bla2 = struct.unpack('d'*unpack_length, bla) #returns a tuple...not an array :(
bla2_array = double(bla2);
bla2_array_image = bla2_array.reshape(mat_shape,order='C')
imshow(bla2_array_image[0])
#read using numpy.fromfile:
bla_np = np.fromfile(file_name,'d',count=read_size);
imshow(bla_np.reshape(ROI_shape));
#using array.array
bla_double_array = array.array('d',bla);
######################################################################################################################################################################################################


######################################################################################################################################################################################################
#ASTROPY:
import astropy.table
import astropy.units as u
import numpy as np
 
# Create table from scratch
ra = np.random.random(5)
t = table.Table()
t.add_column(table.Column(name='ra', data=ra, units=u.degree))
# Write out to file
t.write('myfile.fits')  # also support HDF5, ASCII, etc.
# Read in from file
t = table.Table.read('myfile.fits')
######################################################################################################################################################################################################



######################################################################################################################################################################################################
#PICKLE:
import pickle  # or import cPickle as pickle
# Create dictionary, list, etc.
favorite_color = { "lion": "yellow", "kitty": "red" }
bla = {'blaa': randn(5,5)};
# Write to file
f_myfile = open('myfile.pickle', 'wb')
pickle.dump(favorite_color, f_myfile)
pickle.dump(bla, f_myfile);
f_myfile.close()
# Read from file
f_myfile = open('myfile.pickle', 'rb')
favorite_color2 = pickle.load(f_myfile)  # variables come out in the order you put them in
bla_2 = pickle.load(f_myfile)
f_myfile.close()

bla_2.keys();
values = bla_2.get(bla_2.keys());


class Company(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
company1 = Company('banana', 40)
import dill
import pickle
dill.detect.trace(True)
with open('company_dill.pkl', 'wb') as f:
     dill.dump(company1, f)
     pickle.dump(company1, f, pickle.HIGHEST_PROTOCOL)



#KLEPTO:
#(1). SAVING:
from klepto.archives import file_archive
db = file_archive('foo.txt')
db['1'] = 1
db['max'] = 'bla'
squared = lambda x: x**2
db['squared'] = squared
def add(x,y):
  return x+y

db['add'] = add
class Foo(object):
  y = 1
  def bar(self, x):
    return self.y + x

db['Foo'] = Foo
f = Foo()
db['f'] = f  
db.dump()


#(2). LOADING:
from klepto.archives import file_archive
db = file_archive('foo.txt')
db
file_archive('foo.txt', {}, cached=True)
db.load()
db
file_archive('foo.txt', {'1': 1, 'add': <function add at 0x10610a0c8>, 'f': <__main__.Foo object at 0x10510ced0>, 'max': <built-in function max>, 'Foo': <class '__main__.Foo'>, 'squared': <function <lambda> at 0x10610a1b8>}, cached=True)
db['add'](2,3)
5
db['squared'](3)
9
db['f'].bar(4)
5


#Klepto template:
#(1). Create dataset:
from klepto.archives import file_archive
file_name = 'dataset.txt';
db = file_archive(file_name)
db['1'] = 1
db['max'] = 'bla'
squared = lambda x: x**2
db['squared'] = squared
#(2). Save:
db.dump();
#(3). Load when wanted:
db = file_archive(file_name);
db.load();
#(4). Access dataset parameters:
parameter_keys_names_list = list(db.keys());
parameter_values_list = list(db.values());
parameter_dictionary = dict( zip( parameter_keys_names_list,parameter_values_list));


######################################################################################################################################################################################################



######################################################################################################################################################################################################
# JSON:
# -*- coding: utf-8 -*-
import json

# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Define data
data = {'a list': [1, 42, 3.141, 1337, 'help', u'€'],
        'a string': 'bla',
        'another dict': {'foo': 'bar',
                         'key': 'value',
                         'the answer': 42}}

# Write JSON file
with io.open('data.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(training_generator_object,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(to_unicode(str_))

# Read JSON file
with open('data.json') as data_file:
    data_loaded = json.load(data_file)

print(data == data_loaded)
Explanation of the parameters of json.dump:

indent: Use 4 spaces to indent each entry, e.g. when a new dict is started (otherwise all will be in one line),
sort_keys: sort the keys of dictionaries. This is useful if you want to compare json files with a diff tool / put them under version control.
separators: To prevent Python from adding trailing whitespaces
######################################################################################################################################################################################################



######################################################################################################################################################################################################
# CSV:
import csv

# Define data
data = [(1, "A towel,", 1.0),
        (42, " it says, ", 2.0),
        (1337, "is about the most ", -1),
        (0, "massively useful thing ", 123),
        (-2, "an interstellar hitchhiker can have.", 3)]

# Write CSV file
with open('test.csv', 'w') as fp:
    writer = csv.writer(fp, delimiter=',')
    # writer.writerow(["your", "header", "foo"])  # write header
    writer.writerows(data)

# Read CSV file
with open('test.csv', 'r') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    # next(reader, None)  # skip the headers
    data_read = [row for row in reader]

print(data_read)
######################################################################################################################################################################################################


######################################################################################################################################################################################################
# HD5F
import h5py
import numpy as np

#Create an array:
X,y = get_speckle_sequences();


#### WRITE: ######
#Create h5 file:
file = h5py.File('dset.h5','w')

# Create a dataset under the Root group.
dataset = file.create_dataset("dset",(4, 6), h5py.h5t.STD_I32BE)
print("Dataset dataspace is", dataset.shape)
print("Dataset Numpy datatype is", dataset.dtype)
print("Dataset name is", dataset.name)
print("Dataset is a member of the group", dataset.parent)
print("Dataset was created in the file", dataset.file)

# OR:  
# Create a dataset under the Root group.
comp_type = np.dtype([('Orbit', 'i'), ('Location', np.str_, 6), ('Temperature (F)', 'f8'), ('Pressure (inHg)', 'f8')])
dataset = file.create_dataset("DSC",(4,), comp_type) #CAN I DO THIS TWICE?
data = np.array([(1153, "Sun   ", 53.23, 24.57), (1184, "Moon  ", 55.12, 22.95), (1027, "Venus ", 103.55, 31.23), (1313, "Mars  ", 1252.89, 84.11)], dtype = comp_type)
dataset[...] = data #????

# OR:
# Create "IntArray" dataset.
dim0 = 8
dim1 = 10
dataset = file.create_dataset("IntArray", (dim0,dim1), "i")
data = np.zeros((dim0, dim1))
dataset[...] = data #????
print("Data written to file:")
print(dataset[...])

# OR (APPENDING DATA):
# Create /DS1 dataset; in order to use compression, dataset has to be chunked.
dataset = file.create_dataset('DS1',(4,7),'i',chunks=(3,3), maxshape=(None, None)) 
# Initialize data.
data = np.zeros((4,7))
# Write data.
print ("Writing data...")
dataset[...] = data
# Add two rows filled with 1
dataset.resize((6,7))
dataset[4:6] = 1
# Add three columns filled with 2 
dataset.resize((6,10))
dataset[:,7:10] = 2 
data = dataset[...]
print("Data after extension: ")
print(data)



#### READ: ######
# Open an existing file using defaut properties.
h5_file = h5py.File('dset.h5','r+')
# Open "dset" dataset.
dataset = h5_file['/dset']


# Create string attribute - NOTICE THIS IS ADDING/WRITING DATA
attr_string = "Meter per second"
dataset.attrs["Units"] = attr_string #is this creating(!) an attribute called units like at the bottom?
# Create integer array attribute.
attr_data = np.zeros((2))
attr_data[0] = 100
attr_data[1] = 200
dataset.attrs.create("Speed", attr_data, (2,), h5py.h5t.STD_I32BE)


# Initialize buffers,read and print data.
# Python float type is 64-bit, one needs to use NATIVE_DOUBLE HDF5 type to read data. 
data_read64 = np.zeros((4,6,), dtype=float) #insert data to data_read64???
dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, data_read64, mtype=h5py.h5t.NATIVE_DOUBLE) 
print "Printing data 64-bit floating numbers..."
print data_read64
# Now using float32:
data_read32 = np.zeros((4,6,), dtype=np.float32)
dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, data_read32, mtype=h5py.h5t.NATIVE_FLOAT)
print "Printing data 32-bit floating numbers..."
print data_read32


# This example shows how to read a hyperslab from an existing dataset.
# Open file and read dataset.
file = h5py.File('hype.h5', 'r')
dataset = file['IntArray']
data_in_file = dataset[...]
print "Data in file ..."
print data_in_file[...]
# Initialize data with 0s.
data_selected = np.zeros((8,10), dtype=np.int32)
# Read selection.
space_id = dataset.id.get_space()
space_id.select_hyperslab((1,1), (2,2), stride=(4,4), block=(2,2))
#---> Doesn't work dataset.id.read(space_id, space_id, data_selected, h5py.h5t.STD_I32LE) 
dataset.id.read(space_id, space_id, data_selected) 
print("Seleted data read from file....")
print(data_selected[...])


#Resizing h5df:
with h5py.File('.\PreprocessedData.h5', 'a') as hf:
    hf["X_train"].resize((hf["X_train"].shape[0] + X_train_data.shape[0]), axis = 0)
    hf["X_train"][-X_train_data.shape[0]:] = X_train_data

    hf["X_test"].resize((hf["X_test"].shape[0] + X_test_data.shape[0]), axis = 0)
    hf["X_test"][-X_test_data.shape[0]:] = X_test_data

    hf["Y_train"].resize((hf["Y_train"].shape[0] + Y_train_data.shape[0]), axis = 0)
    hf["Y_train"][-Y_train_data.shape[0]:] = Y_train_data

    hf["Y_test"].resize((hf["Y_test"].shape[0] + Y_test_data.shape[0]), axis = 0)
    hf["Y_test"][-Y_test_data.shape[0]:] = Y_test_data
    

######################################################################################################################################################################################################
    
    
    




######################################################################################################################################################################################################

# Open an existing file using defaut properties.
h5_file = h5py.File('dset.h5','r+')

# Open "dset" dataset.
dataset = h5_file['/dset']

# Create string attribute.
attr_string = "Meter per second"
dataset.attrs["Units"] = attr_string

# Create integer array attribute.
attr_data = np.zeros((2))
attr_data[0] = 100
attr_data[1] = 200

dataset.attrs.create("Speed", attr_data, (2,), h5py.h5t.STD_I32BE)
# Close the file before exiting

h5_file.close();






# This examaple creates an HDF5 file dset.h5 and an empty datasets /dset in it.

# Create a new file using defaut properties.
file = h5py.File('dset.h5','w')

# Create a dataset under the Root group.
dataset = file.create_dataset("dset",(4, 6), h5py.h5t.STD_I32BE)
print "Dataset dataspace is", dataset.shape
print "Dataset Numpy datatype is", dataset.dtype
print "Dataset name is", dataset.name
print "Dataset is a member of the group", dataset.parent
print "Dataset was created in the file", dataset.file

# Close the file before exiting
file.close()



# This example reads integer data from dset.h5 file into Python floatng buffers.

# Open an existing file using default properties.
file = h5py.File('dset.h5','r+')

# Open "dset" dataset under the root group.
dataset = file['/dset']

# Initialize buffers,read and print data.
# Python float type is 64-bit, one needs to use NATIVE_DOUBLE HDF5 type to read data. 
data_read64 = np.zeros((4,6,), dtype=float)
dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, data_read64, mtype=h5py.h5t.NATIVE_DOUBLE)
print "Printing data 64-bit floating numbers..."
print data_read64

data_read32 = np.zeros((4,6,), dtype=np.float32)
dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, data_read32, mtype=h5py.h5t.NATIVE_FLOAT)
print "Printing data 32-bit floating numbers..."
print data_read32

# Close the file before exiting
file.close()





# This example creates an HDF5 file compound.h5 and an empty datasets /DSC in it.

# Create a new file using default properties.
file = h5py.File('compound.h5','w')

# Create a dataset under the Root group.
comp_type = np.dtype([('Orbit', 'i'), ('Location', np.str_, 6), ('Temperature (F)', 'f8'), ('Pressure (inHg)', 'f8')])
dataset = file.create_dataset("DSC",(4,), comp_type)
data = np.array([(1153, "Sun   ", 53.23, 24.57), (1184, "Moon  ", 55.12, 22.95), (1027, "Venus ", 103.55, 31.23), (1313, "Mars  ", 1252.89, 84.11)], dtype = comp_type)
dataset[...] = data

# Close the file before exiting
file.close()
file = h5py.File('compound.h5', 'r')
dataset = file["DSC"]
print "Reading Orbit and Location fields..."
orbit = dataset['Orbit']
print "Orbit: ", orbit
location = dataset['Location']
print "Location: ", location
data = dataset[...]
print "Reading all records:"
print data
print "Second element of the third record:", dataset[2, 'Location']
file.close()





# This example shows how to write a hyperslab to an existing dataset.
# Create a file using default properties.
file = h5py.File('hype.h5','w')

# Create "IntArray" dataset.
dim0 = 8
dim1 = 10
dataset = file.create_dataset("IntArray", (dim0,dim1), "i")

# Initialize data object with 0.
data = np.zeros((dim0, dim1))

# Initialize data for writing.
for i in range(dim0):
    for j in range(dim1):
        if j < dim1/2: 
            data[i][j]= 1
        else:
            data[i][j] = 2 	 

# Write data
dataset[...] = data
print "Data written to file:"
print dataset[...]

# Close the file before exiting
file.close()

# Open the file and dataset.
file = h5py.File('hype.h5','r+')
dataset = file['IntArray']

# Write a selection.
dataset[1:4, 2:6] = 5
print "Data after selection is written:"
print dataset[...]

# Close the file before exiting
file.close()







# This example shows how to read a hyperslab from an existing dataset.

# Open file and read dataset.
file = h5py.File('hype.h5', 'r')
dataset = file['IntArray']
data_in_file = dataset[...]
print "Data in file ..."
print data_in_file[...]

# Initialize data with 0s.
data_selected = np.zeros((8,10), dtype=np.int32)

# Read selection.
space_id = dataset.id.get_space()
space_id.select_hyperslab((1,1), (2,2), stride=(4,4), block=(2,2))
#---> Doesn't work dataset.id.read(space_id, space_id, data_selected, h5py.h5t.STD_I32LE) 
dataset.id.read(space_id, space_id, data_selected) 
print "Seleted data read from file...."
print data_selected[...]

# Close the file before exiting
file.close()




# This example demonstrates how to do point selection in Python.
file1 = h5py.File('copy1.h5','w')
file2 = h5py.File('copy2.h5','w')
dataset1 = file1.create_dataset('Copy1', (3,4), 'i')
dataset2 = file2.create_dataset('Copy2', (3,4), 'i')

# Initialize data object with 0.
data1 = np.zeros((3,4))
data2 = np.ones((3,4))
val = (55,59)

# Write data
dataset1[...] = data1
dataset2[...] = data2

# Modify two elements with the new values. We can choose any number of elements along one dimension.
dataset1[0, [1,3]] = val
dataset2[0, [1,3]] = val
file1.close()
file2.close()

# Reopen the files  and read data back
file1 = h5py.File('copy1.h5', 'r')
dataset1 = file1['Copy1']
data1 = dataset1[...]
print "Dataset Copy1 in copy1.h5:"
print data1

file2 = h5py.File('copy2.h5', 'r')
dataset2 = file2['Copy2']
data2 = dataset2[...]
print "Dataset Copy2 in copy2.h5:"
print data2

file1.close()
file2.close()






# This example creates a dataset and then extends it by rows and then by columns.

# Create  unlim.h5 file.
file = h5py.File('unlim.h5','w')

# Create /DS1 dataset; in order to use compression, dataset has to be chunked.
dataset = file.create_dataset('DS1',(4,7),'i',chunks=(3,3), maxshape=(None, None)) 

# Initialize data.
data = np.zeros((4,7))

# Write data.
print "Writing data..."
dataset[...] = data
file.close()

# Read data back; display compression properties and dataset max value. 
file = h5py.File('unlim.h5','r+')
dataset = file['DS1']
data = dataset[...]
print "Data before extension: "
print data

# Add two rows filled with 1
dataset.resize((6,7))
dataset[4:6] = 1

# Add three columns filled with 2 
dataset.resize((6,10))
dataset[:,7:10] = 2 
data = dataset[...]
print "Data after extension: "
print data
file.close()




import h5py
filename = 'file.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])



import os
import h5py
import numpy as np
path = '/tmp/out.h5'
os.remove(path)
with h5py.File(path, "a") as f:
    dset = f.create_dataset('voltage284', (10**5,), maxshape=(None,),
                            dtype='i8', chunks=(10**4,))
    dset[:] = np.random.random(dset.shape)        
    print(dset.shape)
    # (100000,)

    for i in range(3):
        dset.resize(dset.shape[0]+10**4, axis=0)   
        dset[-10**4:] = np.random.random(10**4)
        print(dset.shape)
        # (110000,)
        # (120000,)
        # (130000,)



#Resizing h5df:
with h5py.File('.\PreprocessedData.h5', 'a') as hf:
    hf["X_train"].resize((hf["X_train"].shape[0] + X_train_data.shape[0]), axis = 0)
    hf["X_train"][-X_train_data.shape[0]:] = X_train_data

    hf["X_test"].resize((hf["X_test"].shape[0] + X_test_data.shape[0]), axis = 0)
    hf["X_test"][-X_test_data.shape[0]:] = X_test_data

    hf["Y_train"].resize((hf["Y_train"].shape[0] + Y_train_data.shape[0]), axis = 0)
    hf["Y_train"][-Y_train_data.shape[0]:] = Y_train_data

    hf["Y_test"].resize((hf["Y_test"].shape[0] + Y_test_data.shape[0]), axis = 0)
    hf["Y_test"][-Y_test_data.shape[0]:] = Y_test_data

######################################################################################################################################################################################################





######################################################################################################################################################################################################
### JSON: ###
    
    
######################################################################################################################################################################################################    





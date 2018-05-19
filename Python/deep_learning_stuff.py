# DEEP LEARNING STUFF COLLECTED 

# Change learning rate directly:
vgg.model.optimizer.lr = 0.01



# Showing classification results in an informative way:
#1. A few correct labels at random
correct = np.where(preds==val_labels[:,1])[0]
idx = permutation(correct)[:n_view]
plots_idx(idx, probs[idx])
#2. A few incorrect labels at random
incorrect = np.where(preds!=val_labels[:,1])[0]
idx = permutation(incorrect)[:n_view]
plots_idx(idx, probs[idx])
#3. The images we most confident were cats, and are actually cats
correct_cats = np.where((preds==0) & (preds==val_labels[:,1]))[0]
most_correct_cats = np.argsort(probs[correct_cats])[::-1][:n_view]
plots_idx(correct_cats[most_correct_cats], probs[correct_cats][most_correct_cats])
# as above, but dogs
correct_dogs = np.where((preds==1) & (preds==val_labels[:,1]))[0]
most_correct_dogs = np.argsort(probs[correct_dogs])[:n_view]
plots_idx(correct_dogs[most_correct_dogs], 1-probs[correct_dogs][most_correct_dogs])
#3. The images we were most confident were cats, but are actually dogs
incorrect_cats = np.where((preds==0) & (preds!=val_labels[:,1]))[0]
most_incorrect_cats = np.argsort(probs[incorrect_cats])[::-1][:n_view]
plots_idx(incorrect_cats[most_incorrect_cats], probs[incorrect_cats][most_incorrect_cats])
#3. The images we were most confident were dogs, but are actually cats
incorrect_dogs = np.where((preds==1) & (preds!=val_labels[:,1]))[0]
most_incorrect_dogs = np.argsort(probs[incorrect_dogs])[:n_view]
plots_idx(incorrect_dogs[most_incorrect_dogs], 1-probs[incorrect_dogs][most_incorrect_dogs])
#5. The most uncertain labels (ie those with probability closest to 0.5).
most_uncertain = np.argsort(np.abs(probs-0.5))
plots_idx(most_uncertain[:n_view], probs[most_uncertain])
#Confusion matrix
cm = confusion_matrix(val_classes, preds)
plot_confusion_matrix(cm, val_batches.class_indices)



######## USING FFT #######
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = img.shape
crow,ccol = rows/2 , cols/2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()




###### FFT AND DFT IN OPENCV (RETURNS REAL AND IMAGINARY VERSUS COMPLEX) #######
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# Note You can also use cv2.cartToPolar() which returns both magnitude and phase in a single shot



######## DFT OPTIMIZATION IN OPENCV ######
img = cv2.imread('messi5.jpg',0)
rows,cols = img.shape


nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
print nrows, ncols

#See, the size (342,548) is modified to (360, 576). Now let’s pad it with zeros (for OpenCV) 
#and find their DFT calculation performance. 
#You can do it by creating a new big zero array and copy the data to it, or use cv2.copyMakeBorder().

nimg = np.zeros((nrows,ncols))
nimg[:rows,:cols] = img
#OR:

right = ncols - cols
bottom = nrows - rows
bordertype = cv2.BORDER_CONSTANT #just to avoid line breakup in PDF file
nimg = cv2.copyMakeBorder(img,0,bottom,0,right,bordertype, value = 0)
Now we calculate the DFT performance comparison of Numpy function:

#In [22]: %timeit fft1 = np.fft.fft2(img)
#10 loops, best of 3: 40.9 ms per loop
#In [23]: %timeit fft2 = np.fft.fft2(img,[nrows,ncols])
#100 loops, best of 3: 10.4 ms per loop
#It shows a 4x speedup. Now we will try the same with OpenCV functions.
#
#In [24]: %timeit dft1= cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
#100 loops, best of 3: 13.5 ms per loop
#In [27]: %timeit dft2= cv2.dft(np.float32(nimg),flags=cv2.DFT_COMPLEX_OUTPUT)
#100 loops, best of 3: 3.11 ms per loop


    
    
    
############## ARG PARSE !!! #######
# arg.py

import argparse
import sys

def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Script to learn basic argparse')
    parser.add_argument('-H', '--host',
                        help='host ip',
                        required='True',
                        default='localhost')
    parser.add_argument('-p', '--port',
                        help='port of the web server',
                        default='8080')
    parser.add_argument('-u', '--user',
                        help='user name',
                        default='root')

    results = parser.parse_args(args)
    return (results.host,
            results.port,
            results.user)

if __name__ == '__main__':
    h, p, u = check_arg(sys.argv[1:])
    print 'h =',h
    print 'p =',p
    print 'u =',
    
#If we run it:
#
#$ python arg.py -H 192.17.23.5
#h = 192.17.23.5
#p = 8080
#u = root


## usin help:
#$ python arg.py -h
#usage: arg.py [-h] -H HOST [-p PORT] [-u USER]
#
#Script to learn basic argparse
#
#optional arguments:
#  -h, --help            show this help message and exit
#  -H HOST, --host HOST  host ip
#  -p PORT, --port PORT  port of the web server
#  -u USER, --user USER  user name
#Notice that we used -H for host-ip mandatory option instead of lower case 'h' 
#because it is reserved for 'help.



####### ITERTAORS ####
    >>> # Python 3
>>> iterable = [1,2,3]
>>> iterator = iterable.__iter__()    # or iterator = iter(iterable)
>>> type(iterator)
<type 'listiterator'>
>>> value = iterator.__next__()   # or value = next(iterator)
>>> print(value)
1
>>> value = next(iterator)
>>> print(value)
2
>>>
>>> # Python 2
>>> iterable = [1,2,3]
>>> iterator = iterable.__iter__()
>>> type(iterator)
<type 'listiterator'>
>>> value = iterator.next()
>>> value
1
>>> value = next(iterator)
>>> value
2


#####
def foo_with_yield():
    yield 1
    yield 2
    yield 3

# iterative calls
for yield_value in foo_with_yield():
    print yield_value,
Output

1 2 3


######
def foo_with_yield():
    yield 1
    yield 2
    yield 3

x=foo_with_yield()
print x
print next(x)
print x
print next(x)
print x
print next(x)
Output:

<generator object foo_with_yield at 0x7f6e4f0f1e60>
1
<generator object foo_with_yield at 0x7f6e4f0f1e60>
2
<generator object foo_with_yield at 0x7f6e4f0f1e60>
3




########### ZIP and LIST COMPREHENSION... VERY IMPORTANT #######
>>> a = [1,2,3]
>>> b = [4,5,6]
#ZIP!!!
>>> z = list(zip(a,b))
>>> z
[(1, 4), (2, 5), (3, 6)]
#UNZIP!!!!
>>> c, d = zip(*z)
>>> c, d
((1, 2, 3), (4, 5, 6))



>>> keys = ['a', 'b', 'c']
>>> values = [1, 2, 3]
>>> list(zip(keys,values))
[('a', 1), ('b', 2), ('c', 3)]

>>> D2 = {}
>>> for (k,v) in zip(keys, values):
...     D2[k] = v
... 
>>> D2
{'a': 1, 'b': 2, 'c': 3}
>>> D3 = dict(zip(keys, values))
>>> D3
{'a': 1, 'b': 2, 'c': 3}



#initializing a dictionary:
When we want initialize a dict from keys, we do this:

>>> D = dict.fromkeys(['a','b','c'], 0)
>>> D
{'a': 0, 'c': 0, 'b': 0}

We can use dictionary comprehension to do the same thing;

>>> D = {k: 0 for k in ['a','b','c']}
>>> D
{'a': 0, 'c': 0, 'b': 0}



#CONDITIONAL ZIP!!!!
Conditional zip()
>>> x = [1,2,3,4,5]
>>> y = [11,12,13,14,15]
>>> condition = [True,False,True,False,True]
>>> [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]
[1, 12, 3, 14, 5]

#USING NUMPY WHERE:
The same thing can be done using NumPy's where:
>>> import numpy as np
>>> np.where([1,0,1,0,1], np.arange(1,6), np.arange(11,16))
array([ 1, 12,  3, 14,  5])





###### LIST FILTERING!!!! ########
>>> 
>>> [x for x in range(10) if x % 2 == 0]
[0, 2, 4, 6, 8]

>>> list(filter((lambda x: x % 2 == 0), range(10)))
[0, 2, 4, 6, 8]

>>> result = []
>>> for x in range(10):
	if x % 2 == 0:
		result.append(x)
	
>>> result
[0, 2, 4, 6, 8]
>>> 


>>> list( map((lambda x: x ** 2), filter((lambda x: x % 2== 0),range(10))) )
[0, 4, 16, 36, 64]


### NESTED LOOPS EQUIVALENT FOR LIST COMPREHENSION!!!!! #####
>>> result = []
>>> result = [ x ** y for x in [10, 20, 30] for y in [2, 3, 4]]
>>> result
[100, 1000, 10000, 400, 8000, 160000, 900, 27000, 810000]
>>> 
More verbose version is:

>>> result = []
>>> for x in [10, 20, 30]:
	for y in [2, 3, 4]:
		result.append(x ** y)

		
>>> result
[100, 1000, 10000, 400, 8000, 160000, 900, 27000, 810000]
>>> 



Though list comprehensions construct lists, they can iterate over any sequence:

>>> [x + y for x in 'ball' for y in 'boy']
['bb', 'bo', 'by', 'ab', 'ao', 'ay', 'lb', 'lo', 'ly', 'lb', 'lo', 'ly']
>>> 
Here is a much more complicated list comprehension example:

>>> [(x,y) for x in range(5) if x % 2 == 0 for y in range(5) if y % 2 == 1]
[(0, 1), (0, 3), (2, 1), (2, 3), (4, 1), (4, 3)]
The expression permutes even numbers from 0 through 4 with odd numbers from 0 through 4. Here is an equivalent version which is much more verbose:

>>> result = []
>>> for x in range(5):
	if x % 2 == 0:
		for y in range(5):
			if y % 2 == 1:
				result.append((x,y))

				
>>> result
[(0, 1), (0, 3), (2, 1), (2, 3), (4, 1), (4, 3)]
>>> 



######## USING THE MAP OPERATOR ######
The map(aFunction, aSequence) function applies a passed-in function to each item in an 
iterable object and returns a list containing all the function call results.

>>> items = [1, 2, 3, 4, 5]
>>> 
>>> def sqr(x): return x ** 2

>>> list(map(sqr, items))
[1, 4, 9, 16, 25]
>>> 

We passed in a user-defined function applied to each item in the list. map calls sqr on each 
list item and collects all the return values into a new list.



x = [1,2,3]
y = [4,5,6]

from operator import add #!!!!!!!!!!!!!!!!!!!! LOOK AT THE DIFFERENT OPERATORS THEY CAN COME HANDY!!!!!!
print map(add, x, y)  # output [5, 7, 9]



####### FILTERING SEQUENCES!!!!! #######
>>> list(range(-5,5))
[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
>>>
>>> list( filter((lambda x: x < 0), range(-5,5)))
[-5, -4, -3, -2, -1]
>>> 



a = [1,2,3,5,7,9]
b = [2,3,5,6,7,8]
print filter(lambda x: x in a, b)  # prints out [2, 3, 5, 7]
Note that we can do the same with list comprehension:
a = [1,2,3,5,7,9]
b = [2,3,5,6,7,8]
print [x for x in a if x in b] # prints out [2, 3, 5, 7]




##################### SCIKIT-IMAGE (SKIMAGE) #############################
I/O: skimage.io
>>>
>>> from skimage import io
Reading from files: skimage.io.imread()
>>>
>>> import os
>>> filename = os.path.join(skimage.data_dir, 'camera.png')
>>> camera = io.imread(filename)


logo = io.imread('http://scikit-image.org/_static/img/logo.png')
Saving to files:
>>>
>>> io.imsave('local_logo.png', logo)


from skimage import img_as_float
>>> camera_float = img_as_float(camera)
>>> camera.max(), camera_float.max()
(255, 1.0)


from skimage import morphology
morphology.diamond(1)


a = np.zeros((7,7), dtype=np.uint8)
>>> a[1:6, 2:5] = 1
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
>>> morphology.binary_erosion(a, morphology.diamond(1)).astype(np.uint8)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
>>> #Erosion removes objects smaller than the structure
>>> morphology.binary_erosion(a, morphology.diamond(2)).astype(np.uint8)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    
    
    
    a = np.zeros((5, 5))
>>> a[2, 2] = 1
>>> a
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> morphology.binary_dilation(a, morphology.diamond(1)).astype(np.uint8)
array([[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0]], dtype=uint8)

    
    
    
    
    
    a = np.zeros((5,5), dtype=np.int)
>>> a[1:4, 1:4] = 1; a[4, 4] = 1
>>> a
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 1]])
>>> morphology.binary_opening(a, morphology.diamond(1)).astype(np.uint8)
array([[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0]], dtype=uint8)

    
    
######### TURN AN IMAGE INTO A 1 ELEMENT BATCH:
im_Array = np.expand_dims(im_Array,axis=0)
    
######### PLAY WITH TENSORS DIMENSIONS ORDERING:
im_array = imarray.transpose((2,0,1))   





##### TRAINING LARGE SET - FLOW FROM DIRECTORY ##########
train_on_batch

train_on_batch(self, x, y, class_weight=None, sample_weight=None)
It's designed to perform a single gradient update over one batch of samples.

Custom Generator

Another idea is to use generators which provide you with data given a directory. They can also be used for data augmentation, i.e. randomly generating new training data from your data. Here is an example from the keras documentation:

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# just rescale test data
test_datagen = ImageDataGenerator(rescale=1./255)

# this generator loads data from the given directory and 32 images 
# chunks called batches. you can set this as you like
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# same es the train_generator    
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# loads sequentially images and feeds them to the model. 
# the batch size is set in the constructor 
model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)







############### USING CUSTOM DATA GENERATORS (DOING IT RIGHT) !!!!! #########
import numpy as np

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 32, dim_y = 32, dim_z = 32, batch_size = 32, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, labels, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(labels, list_IDs_temp)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, labels, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
      y = np.empty((self.batch_size), dtype = int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
          X[i, :, :, :, 0] = np.load(ID + '.npy')

          # Store class
          y[i] = labels[ID]

      return X, sparsify(y)

def sparsify(y):
  'Returns labels in binary NumPy array'
  n_classes = # Enter number of classes
  return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])
    




import numpy as np

from keras.models import Sequential
from my_classes import DataGenerator

# Parameters
params = {'dim_x': 32,
          'dim_y': 32,
          'dim_z': 32,
          'batch_size': 32,
          'shuffle': True}

# Datasets
partition = # IDs
labels = # Labels

# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(partition['train'])//batch_size,
                    validation_data = validation_generator,
                    validation_steps = len(partition['validation'])//batch_size)





####### IMPORTING DATA USING MULTIPLE WORKERS!!! ######
import threading

class threadsafe_iter(object):
  """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
  def __init__(self, it):
      self.it = it
      self.lock = threading.Lock()

  def __iter__(self):
      return self

  def __next__(self):
      with self.lock:
          return self.it.__next__()

def threadsafe_generator(f):
  """
    A decorator that takes a generator function and makes it thread-safe.
    """
  def g(*a, **kw):
      return threadsafe_iter(f(*a, **kw))
  return g
#Now, let's import the threadsafe_generator function at the beginning of the my_classes.py script:

from tools import threadsafe_generator
#Next, locate the generate method of the DataGenerator class. A
#pply on it the threadsafe_generator decorator just like shown below.

@threadsafe_generator
def generate(self, labels, list_IDs):
  [...] # Code    
    



#Tell Keras to use multiple cores

#Congratulations, you have done the most difficult part! 
#The only thing left to do is to add one argument in the fit_generator call of your script, 
#in which you will tell Keras how many cores you would like to use for real-time data generation.

#Provided that you have specified your desired number of workers in n_workers, 
#the fit_generator call will become

model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(partition['train'])//batch_size,
                    validation_data = validation_generator,
                    validation_steps = len(partition['validation'])//batch_size,
                    workers = n_workers)




###### ANOTHER GENERATOR FROM FILE READING LINES (MAYBE SUITABLE FOR BINARY)  ######
def generate_arrays_from_file(path):
    while 1:
        f = open(path);
        for line in f:
            #create numpy arrays of input data and abels from each line in the file
            [x1,x2,y] = process_line(line)
            yield ({'input1':x1,'input2':x2},{'output':y})
        f.close()




############ SAVING MODEL ARCHITECTURE AND MODEL WEIGHTS #############
from keras.models import load_model
model.save('my_model.h5');
del model
model = load_model('my_model.h5');

model.save_weights('my_model_weights.h5');
model.load_weights('my_model_weights.h5');

#### VERY IMPORTANT!!!- if i want to load weight into a DIFFERENT ARCHITECTURE (with some layers in common)
#for instance for fine-tuning or transfer-learning, you can loa weights by layer name:
model.load_weights('my_model_weights.h5', by_name=True);

#### VERY IMPORTANT!!!- if i have a custom class or function or layer, i can pass them to the loading
#mechanism via the custm_objects argument:
model = load_model('my_model.h5', custom_objects={'AttentionLayer':AttentionLayer})


#IF I ONLY WANT TO SAVE MODEL ARCHITECTURE (NOT THE WEIGHTS)
json_string = model.to_json()
model = model_from_json(json_string)






############## MODEL VISUALIZATION !!!!!!!!!!!!!!!!!!!!!! ######
from keras.utils import plot_model
plot_model(model, to_file='model.png')

fom IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot',format='svg'))


#####  ABOUT KERAS LAYERS ######
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)







###################### TENSORFLOW WITH KERAS ###############################
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
#IS THIS PLACEHOLDER' SHAPE DEFINED IN A DIFFERENT WAY THAN IN KERAS??!?!?!
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

# Keras layers can be called on TensorFlow tensors:
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})
    
from keras.metrics import categorical_accuracy as accuracy
acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels})




###################### VIZUALIZE WEIGHTS ###############################

#for a 2 layer model (GENERALIZE TO ANY LAYER):
w1,b1,w2,b2 = model.get_weights()

#let's say the first layer is 128 filters:
sx,sy = (16,8); #16*8=128

f,con = plt.subplots(sx,sy,sharex='col',sharey='row');

for xx in arange(sx):
    for yy in arange(sy):
        con[xx,yy].pcolormesh(w1,[:,sy*xx+yy].reshap(28*28));
        


###################### TEST CUSTOM LAYER ###############################
def test_layer(my_layer,x):
    layer_config = my_layer.get_config()
    layer_config["input_shape"] = x.shape;
    my_layer = my_layer._class_.from_config(layer_config);
    model = Sequential()
    model.add(my_layer);
    model.compile("rmsprop","mse");
    x_ = np.expand_dims(x, axis=0);
    return model.predict(x_)[0]

x = np.random.randn(10,10);
my_layer = Reshape((5,20));
y = test_layer(my_layer,x);
assert(y.shape == (5,20));



################### TIC TOC LIKE IN MATLAB #################
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
    
tic()
time.sleep(5)
toc() # returns "Elapsed time: 5.00 seconds."    
    

#Actually, this is more versatile than the built-in Matlab functions. 
#Here, you could create another instance of the TicTocGenerator to keep track of multiple operations, 
#or just to time things differently. 
#For instance, while timing a script, we can now time each piece of the script seperately, 
#as well as the entire script. (I will provide a concrete example)

TicToc2 = TicTocGenerator() # create another instance of the TicTocGen generator

def toc2(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc2
    tempTimeInterval = next(TicToc2)
    if tempBool:
    print( "Elapsed time 2: %f seconds.\n" %tempTimeInterval )

def tic2():
    # Records a time in TicToc2, marks thebeginning of a time interval
    toc2(False)
    
    
    
    
    
############ VISUALIZE CHANNEL ACTIVATTIONS (BASICALLY THE FILTERED IMAGE FOR DIFFERENT CHANNELS) #################    
from keras import models
# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)    
    
# This will return a list of 8 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()



########### VIZUALIZE THE INPUT IMAGE THAT BEST ACTIVATES ANY FILTER #########
ef generate_pattern(layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
    size = 64
    margin = 5

    # This a empty (black) image where we will store our results.
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()
    
    

############# VIZUALIZE HEATMAP OF CLASS ACTIVATIONS ##########
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# The local path to our target image
img_path = '/Users/fchollet/Downloads/creative_commons_elephant.jpg'

# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

np.argmax(preds[0])    

# This is the "african elephant" entry in the prediction vector
african_elephant_output = model.output[:, 386]
# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')
# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))
# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)


#For visualization purpose, we will also normalize the heatmap between 0 and 1:
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()



import cv2
# We use cv2 to load the original image
img = cv2.imread(img_path)
# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)
# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img
# Save the image to disk
cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)
    
    
    
    
    

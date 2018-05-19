import numpy
from numpy import arange
def my_linspace_int(start,step,number_of_steps):
    stop = start + number_of_steps*step;
    return arange(start,stop,step,int);

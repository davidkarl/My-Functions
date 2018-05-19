import numpy
def my_linspace(start,step,number_of_steps):
    stop = start + number_of_steps*step;
    bla = numpy.arange(start,stop,step);
    return bla


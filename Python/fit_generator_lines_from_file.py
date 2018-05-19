# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 02:23:27 2017

@author: master
"""

def generate_arrays_from_file(path):
    while 1:
    f = open(path)
    for line in f:
        # create numpy arrays of input data
        # and labels, from each line in the file
        x1, x2, y = process_line(line)
        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
    f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        steps_per_epoch=10000, epochs=10)
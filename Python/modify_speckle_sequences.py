# modify speckle sequences
from importlib import reload
import things_to_import
things_to_import = reload(things_to_import)
from things_to_import import *

import get_center_number_of_pixels
from get_center_number_of_pixels import *

                          
def modify_speckle_sequences(X,
                             epp = 10000,
                             readout_noise = 140,
                             background = 0,
                             flag_use_shot_noise = 0,
                             cross_talk = 0, #to all neighbors equally
                             flag_center_each_speckles_independently_or_as_batch = 2, # more accurately - 
                             flag_stretch_each_speckles_independently_or_as_batch = 2,# or_as_sample
                             std_wanted = 1,
                             flag_stateful_batch = 0,
                             stateful_batch_size = 32):
    
    
    #(****). The last two arguments: flag_stateful_batch and stateful_batch_size are for when i would like
    #and if i would like to modify X in a manner which considered the stateful batch and not only the
    #sample.
    #For instance, in the future i may find it more intelligent to modify X batch-wise before predictions.
    
    
    #Get shape paramters:
    number_of_batches = shape(X)[0];
    number_of_time_steps = shape(X)[1];
    ROI = shape(X)[2]; #i assume a square shape
    
    #Multiply to make mean be epp:
    X = X * epp;
    
    #Add noises:
    X += readout_noise * randn(number_of_batches,number_of_time_steps,ROI,ROI,1);
    X += background;
    if flag_use_shot_noise == 1:
        X += sqrt(abs(X)) * randn(number_of_batches,number_of_time_steps,ROI,ROI,1);
    
    
    #Add cross talk:
    #TO DO
    
    
    
    ##################################### MODIFY PER ELEMENT ###########################################
    ### (*). Center Per Samle Element:
    if flag_center_each_speckles_independently_or_as_batch == 1:
        X_mean = mean(X,(2,3),keepdims=True);
        X_mean = repeat(X_mean,ROI,2);
        X_mean = repeat(X_mean,ROI,3);
        X = X - X_mean; #the independent images lie in axes 2 and 3, so i substact the mean
    ## (*). Stretch Per Sample Element:
    if flag_stretch_each_speckles_independently_or_as_batch == 1:
        X_mean = mean(X,(2,3),keepdims=True);
        X_mean = repeat(X_mean,ROI,2);
        X_mean = repeat(X_mean,ROI,3);
        X_centered_std = std(X-X_mean,(2,3),keepdims=True);
        X_centered_std = repeat(X_centered_std,ROI,2);
        X_centered_std = repeat(X_centered_std,ROI,3);
        X = X / X_centered_std * std_wanted;
    #################################################################################################################################
    
    
    
    ################################### MODIFY PER SAMPLE ###################################################################################
    ### (*). Center Per Batch Sample:
    if flag_center_each_speckles_independently_or_as_batch == 2:
        X_mean = mean(X,(1,2,3),keepdims=True);
        X_mean = repeat(X_mean,number_of_time_steps,1);
        X_mean = repeat(X_mean,ROI,2);
        X_mean = repeat(X_mean,ROI,3);
        X = X - X_mean; #the independent images lie in axes 2 and 3, so i substact the mean
    ### (*). Stretch Per Batch Sample:
    if flag_stretch_each_speckles_independently_or_as_batch == 2:
        X_mean = mean(X,(1,2,3),keepdims=True);
        X_mean = repeat(X_mean,number_of_time_steps,1);
        X_mean = repeat(X_mean,ROI,2);
        X_mean = repeat(X_mean,ROI,3);
        X = X - X_mean; #the independent images lie in axes 2 and 3, so i substact the mean
        X_centered_std = std(X-X_mean,(1,2,3),keepdims=True);
        X_centered_std = repeat(X_centered_std,number_of_time_steps,1);
        X_centered_std = repeat(X_centered_std,ROI,2);
        X_centered_std = repeat(X_centered_std,ROI,3);
        X = X / X_centered_std * std_wanted;
    #################################################################################################################################
   
    
    return X




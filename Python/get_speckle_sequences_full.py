from importlib import reload
import things_to_import
things_to_import = reload(things_to_import)
from things_to_import import *

import get_center_number_of_pixels
from get_center_number_of_pixels import *

#Reurns 1 batch of a speckle sequence
#if flag_stateful = 1 then the whole returned sequence is the same speckles and it's of size batch_size
#if flag_stateful = 0 then every sample is a different speckles and batch_size simply means number of samples

#I want this function to be "CLEAN"  - i only create a speckles normalized to 1 with nothing else.
#all other stuff like additive noises and cross talk, centering and stretching should be done outside the function

def get_speckle_sequences_full(ROI=64, 
                          speckle_size = 5, 
                          max_shift = 0.1, 
                          number_of_time_steps = 4, 
                          batch_size = 32, \
                          flag_single_number_or_optical_flow_prediction = 1, \
                          constant_decorrelation_fraction = 0, 
                          Fs = 5500,\
                          flag_stateful_batch = 0, \
                          data_type = 'f'):
    #need to add cross-talk using convolution
    #need to add pixel value clipping

    
    
    #ROI = 64;
    #speckle_size = 15;
    #epp = 10000;
    #readout_noise = 0;
    #optical_SNR = epp/sqrt(epp+readout_noise^2);
    #max_shift = 0.1;
    #number_of_time_steps = 4; 
    #batch_size = 32
    #flag_single_number_or_optical_flow_prediction = 1
    #constant_decorrelation_fraction = 0
    #Fs = 5500;
    #flag_center_each_speckles_independently_or_as_batch = 2; #for no centering press 3
    #flag_stretch_each_speckles_independently_or_as_batch = 2; #for no stretching press 3
    #std_wanted = 1;
    #flag_stateful_batch = 0;
    
    #indexing auxiliary variables
    start = -1;
    end = -1;
    
    #Simulation N:
    N = ROI + 10;
    
    #Get normalization constant which makes speckle intensity mean = 1 (for later epp)
    normalization_constant = sqrt(190.06 * (10/speckle_size)**2 * (N/110)**2);
    
    #Get gaussian beam:
    phase_screen = exp(1j*100*randn(N,N));
    x = arange(-fix(N/2),ceil(N/2),1);
    [X,Y] = meshgrid(x,x);    
    radius = N/speckle_size;
    gaussian_beam = (X**2 + Y**2) < radius;
    gaussian_beam = exp(-(X**2+Y**2)/radius**2);
    
    #Normalize gaussian beam to make speckles intensity image have a mean of 1:
    gaussian_beam = gaussian_beam / normalization_constant;
    gaussian_beam_after_phase = gaussian_beam * phase_screen;
    
    #Get tilt phases k-space:
    delta_f1 = 1/(N);
    f_x = x*delta_f1;
    #Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
    f_x = fftshift(f_x);
    #Build k-space meshgrid:
    [kx,ky] = meshgrid(f_x,f_x); 
    
    
    #############################   TO DO !!!!!   ################################################################
    #I should add the possibility of the label being not only for the middle frame-couple shift, but for
    #every possible shift inside the given batch.... MAYBE LATER.
    
    #I should also add the possibility for pixel-by-pixel shifts (for optical flow prediction) using 
    #something like the MEX version of B-splines like i did in matlab's image warping.
    ###############################################################################################################
    
    
    #Get sequence of movements (only random white noise shifts for now):
    total_number_of_samples = int(batch_size*number_of_time_steps);
    if flag_single_number_or_optical_flow_prediction == 1:
        shifts_x = max_shift*randn(total_number_of_samples); #large enough for any case of stateful
        shifts_y = max_shift*randn(total_number_of_samples);
    else:
        shifts_x = max_shift*randn(total_number_of_samples, N, N);
        shifts_y = max_shift*randn(total_number_of_samples, N, N);
    
    
    #Initialize speckle batch vec (X) and shift labels vec (y):
    number_of_channels = 1; 
    speckles_batch_vec = empty(shape=(int(batch_size),int(number_of_time_steps),int(ROI),int(ROI),int(number_of_channels)));
    if flag_single_number_or_optical_flow_prediction == 1:
        shift_labels_vec = empty(shape=(int(batch_size),2)); #INDEED! SEE HOW TO IMPLEMENT TWO LOSSES (auxiliary loss!!!!!!)
    else:
        shift_labels_vec = empty(shape=(int(batch_size),int(ROI),int(ROI))); #IMPLEMENTING LOSS IN FULLY CONVOLUTIONAL NET!!!!!????
    
    
    #Get shift_labels_vec beforehand:
    shifts_vec_length = batch_size*number_of_time_steps;
    if flag_single_number_or_optical_flow_prediction == 1:
        #SINGLE NUMBER:
        if flag_stateful_batch == 0:
            #In this case every sample has (number_of_time_steps-1) shifts relevant to it.
            #Example: time_steps = 4: then shifts 0,1,2 are for sample 1, and 3,4,5 are for sample 2 etc' ->
            #Output: (1,4,7,10....)
            shift_labels_vec[:,0] = shifts_x[my_linspace_int(start+floor(number_of_time_steps/2),number_of_time_steps-1,batch_size) ];
            shift_labels_vec[:,1] = shifts_y[my_linspace_int(start+floor(number_of_time_steps/2),number_of_time_steps-1,batch_size) ];
        elif flag_stateful_batch == 1:
            #In this case the whole batch is stateful, meaning those are the same speckles.
            #Example: time_steps = 4: then shifts 0,1,2 are for sample 1, and 1,2,3 are for sample 2 etc' ->
            #Output: (1,2,3,4.....)
            shift_labels_vec[:,0] = shifts_x[my_linspace_int(start+floor(number_of_time_steps/2),1,batch_size)];
            shift_labels_vec[:,1] = shifts_y[my_linspace_int(start+floor(number_of_time_steps/2),1,batch_size)];
    else:
        #OPTICAL FLOW:
        1;
    
    
    #Calculate speckle sequences:
    if flag_stateful_batch == 0:
        #Each sample in the batch (which includes number_of_time_steps) from a different speckles:
        
        for batch_counter in arange(0,batch_size,1):
            #make new speckles:
            #(1). Get phase screen:
            phase_screen = exp(1j*100*randn(N,N));
            #(2). put random phase on gaussian beam:
            gaussian_beam_after_phase = gaussian_beam * phase_screen;
            #(3). Create speckles from phase screen:
            speckles = abs(fft2(gaussian_beam_after_phase))**2;
            #(5). Crop center simulation image to get final image:
            speckles = get_center_number_of_pixels(speckles,ROI);
            #(6). Expand dimensions to 3d array (3rd dimension is number of channels):
            speckles = expand_dims(speckles,2); 
            
            #(7). insert first speckles into speckles_batch_vec:
            speckles_batch_vec[batch_counter,0,:,:,:] = speckles;
            
            
            #Get the rest of the shifted speckles:
            for time_counter in arange(0,number_of_time_steps-1,1):
                #get current shifts:
                current_shift_index = batch_counter*(number_of_time_steps-1)+time_counter; 
                shiftx = shifts_x[current_shift_index];
                shifty = shifts_y[current_shift_index];
                
                #get displacement matrix:
                displacement_matrix = exp(-(1j*2*pi*ky*shifty+1j*2*pi*kx*shiftx));  
                
                #get displaced speckles:
                gaussian_beam_after_phase = gaussian_beam_after_phase*displacement_matrix; #the shift or from one image to another, not for the original phase screen
                displaced_speckles = abs(fft2(gaussian_beam_after_phase))**2;                
                displaced_speckles = get_center_number_of_pixels(displaced_speckles,ROI);
                displaced_speckles = expand_dims(displaced_speckles,2);
                
                speckles_batch_vec[batch_counter,time_counter+1,:,:,:] = displaced_speckles;
            #End time_counter loop
        #END batch_counter loop
        
    ###########################################################################################################################################        
        
    elif flag_stateful_batch == 1:
        #the whole batch samples are from 1 original speckle pattern and each sample, including all
        #number_of_time_steps steps are simply shifted by 1 image:
        shift_counter = 0;
        for batch_counter in arange(0,batch_size,1):
            
            if batch_counter == 0:
                #make new speckles which will be common to entire stateful batch:
                #(1). Get phase screen:
                phase_screen = exp(1j*100*randn(N,N));
                #(2). put random phase on gaussian beam:
                gaussian_beam_after_phase = gaussian_beam * phase_screen;
                #(3). Create speckles from phase screen:
                speckles = abs(fft2(gaussian_beam_after_phase))**2;            
                
                #(5). Crop center simulation image to get final image:
                speckles = get_center_number_of_pixels(speckles,ROI);
                #(6). Expand dimensions to 3d array (3rd dimension is number of channels):
                speckles = expand_dims(speckles,2); 

                #(8). insert first speckles into speckles_batch_vec:
                speckles_batch_vec[batch_counter,0,:,:,:] = speckles;
                
                #Renormalize gaussian beam after phase to have speckles mean to be epp:
                for time_counter in arange(1,number_of_time_steps,1):
                    #get current shifts:
                    shiftx = shifts_x[shift_counter];
                    shifty = shifts_y[shift_counter];
                    shift_counter = shift_counter + 1;
                    
                    #get displacement matrix:
                    displacement_matrix = exp(-(1j*2*pi*ky*shifty+1j*2*pi*kx*shiftx));  
                    
                    #get displaced speckles:
                    gaussian_beam_after_phase = gaussian_beam_after_phase*displacement_matrix; #the shift or from one image to another, not for the original phase screen
                    displaced_speckles = abs(fft2(gaussian_beam_after_phase))**2;                    
                
                    displaced_speckles = get_center_number_of_pixels(displaced_speckles,ROI);
                    displaced_speckles = expand_dims(displaced_speckles,2);
                    
                    speckles_batch_vec[batch_counter,time_counter,:,:,:] = displaced_speckles;
                #END time_counter loop
                  
                
            elif batch_counter > 0:
                #Use previous samples for current sample:
                speckles_batch_vec[batch_counter,start+1:-1,:,:,:] = speckles_batch_vec[batch_counter-1,start+2:,:,:,:];
                
                #Just add 1 more shift:
                #get current shifts:
                shiftx = shifts_x[shift_counter];
                shifty = shifts_y[shift_counter];
                shift_counter = shift_counter + 1;
                    
                #get displacement matrix:
                displacement_matrix = exp(-(1j*2*pi*ky*shifty+1j*2*pi*kx*shiftx));  
                
                #get displaced speckles:
                gaussian_beam_after_phase = gaussian_beam_after_phase*displacement_matrix; #the shift or from one image to another, not for the original phase screen
                displaced_speckles = abs(fft2(gaussian_beam_after_phase))**2;
                
                displaced_speckles = get_center_number_of_pixels(displaced_speckles,ROI);
                displaced_speckles = expand_dims(displaced_speckles,2);
                    
                speckles_batch_vec[batch_counter,end,:,:,:] = displaced_speckles;   
                
            #END if batch_counter==0 (time_counter loops)
        
        #END batch_counter loop    
                  
    #END if flag_stateful_batch 
    
     
    return speckles_batch_vec.astype(data_type) , shift_labels_vec.astype(data_type);
    #END get_speckle_sequences()   
    



##Demonstrate:
#import pylab as pl
#for batch_counter in arange(0,3,1):
#    figure()
#    plot_counter = 1;
#    for time_counter in arange(0,number_of_time_steps,1):
#        subplot(1,number_of_time_steps,plot_counter)
#        current_speckles = speckles_batch_vec[batch_counter,time_counter,:,:,0];
#        imshow(current_speckles);
#        title('batch number = '+str(batch_counter)+',\n time counter = '+str(time_counter)
#               + '\n  max = '+str(max(current_speckles)) + ',\n  min = '+str(min(current_speckles)) + '\n '
#               + 'mean = ' + str(mean(current_speckles)) + ',\n std = '+str(std(current_speckles)) );
#        colorbar
#        plot_counter = plot_counter + 1;
#        draw();



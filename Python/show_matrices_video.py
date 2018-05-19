from importlib import reload
import things_to_import
things_to_import = reload(things_to_import)
from things_to_import import *

def show_matrices_video(matrices,time_between_frames, 
                        number_of_batches = 3, 
                        flag_show_min=0, 
                        flag_show_max=0, 
                        flag_show_mean=0, 
                        flag_show_std=0,
                        number_of_decimal_places = 5):
#matrices must be in the form of shape(matrices) = [batch_counter,number_of_time_steps,rows,cols,channels]
    
    batch_size = shape(matrices)[0];
    number_of_time_steps = shape(matrices)[1];
    if number_of_batches != None: batch_size = number_of_batches; 
    
    for batch_counter in arange(0,batch_size,1):
        figure()
        plot_counter = 1;
        for time_counter in arange(0,number_of_time_steps,1):
            subplot(1,number_of_time_steps,plot_counter)
            current_speckles = matrices[batch_counter,time_counter,:,:,0];
            imshow(current_speckles);
            
            title_str = 'batch number = '+str(batch_counter)+',\n time counter = '+str(time_counter);
            if flag_show_min==1: title_str = title_str + '\n  min = ' + str(round(min(current_speckles),number_of_decimal_places))
            if flag_show_max==1: title_str = title_str + '\n  max = ' + str(round(max(current_speckles),number_of_decimal_places))
            if flag_show_mean==1: title_str = title_str + '\n mean = ' + str(round(mean(current_speckles),number_of_decimal_places))
            if flag_show_std==1: title_str = title_str + ',\n std = '+str(round(std(current_speckles),number_of_decimal_places))
            
            title(title_str);
            
            colorbar
            plot_counter = plot_counter + 1;
            draw();






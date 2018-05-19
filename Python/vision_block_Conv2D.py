# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 01:27:31 2017

@author: master
"""

#####################################################################################################################################################################
#####################################################################################################################################################################
########### BASIC CNN LAYERS FOR A SINGLE IMAGE FEATURE EXTRACTION ##############
def vision_block_Conv2D(input_layer, number_of_filters_vec, kernel_size_vec, \
                 flag_dilated_convolution_vec, \
                 flag_resnet_vec, flag_size_1_convolution_on_shortcut_in_resnet_vec, \
                 flag_batch_normalization_vec, \
                 flag_size_1_convolution_after_2D_convolution_vec, flag_batch_normalization_after_size_1_convolution_vec, \
                 activation_type_str_vec):
    1;
    number_of_kernel_sizes_in_layer = length(kernel_size_vec);
    
    for kernel_size_counter in arange(0,number_of_kernel_sizes_in_layer,1):
        #Get parameters for each kernel size filters in layer:
        number_of_filters = number_of_filters_vec[kernel_size_counter];
        kernel_size = kernel_size_vec[kernel_size_counter]; 
        flag_dilated_convolution = flag_dilated_convolution_vec[kernel_size_counter];
        flag_resnet = flag_resnet_vec[kernel_size_counter];
        flag_size_1_convolution_on_shortcut_in_resnet = flag_size_1_convolution_on_shortcut_in_resnet_vec[kernel_size_counter];
        flag_batch_normalization = flag_batch_normalization_vec[kernel_size_counter];
        flag_size_1_convolution_after_2D_convolution = flag_size_1_convolution_after_2D_convolution_vec[kernel_size_counter];
        flag_batch_normalization_after_size_1_convolution = flag_batch_normalization_after_size_1_convolution_vec[kernel_size_counter];
        activation_type_str = activation_type_str_vec[kernel_size_counter];
        
        if flag_dilated_convolution==1:
            vision_block_current_kernel_size = AtrousConvolution2D(number_of_filters, kernel_size, atrous_rate=dilation_rate,border_mode='same')(input_layer);
        else:
            vision_block_current_kernel_size = Conv2D(number_of_filters, kernel_size, padding='same')(input_layer);
            
        if flag_batch_normalization==1:
            vision_block_current_kernel_size = BatchNormalization()(vision_block_current_kernel_size);
        
        vision_block_current_kernel_size = Activation(activation_type_str)(vision_block_current_kernel_size);
            
        if flag_size_1_convolution_after_2D_convolution==1:
            vision_block_current_kernel_size = Conv2D(number_of_filters, 1, padding='same')(vision_block_current_kernel_size);
            if flag_batch_normalization_after_size_1_convolution==1:
                vision_block_current_kernel_size = BatchNormalization()(vision_block_current_kernel_size);
            vision_block_current_kernel_size = Activation(activation_type_str)(vision_block_current_kernel_size);
            
        
        if flag_resnet==1:
            if flag_size_1_convolution_on_shortcut_in_resnet==1:
                input_layer = Conv2D(number_of_filters,1,border_mode='same');    
                if flag_batch_normalization_after_size_1_convolution==1:
                    input_layer = BatchNormalization()(input_layer);
            vision_block_current_kernel_size = Merge([vision_block_current_kernel_size,input_layer],mode='sum')
        
        
        if kernel_size_counter == 0:
            vision_block = vision_block_current_kernel_size;
        else:
            vision_block = Concatenate([vision_block,vision_block_current_kernel_size]);
    #END OF KERNEL SIZE FOR LOOP
       
    return vision_block
#####################################################################################################################################################################
#####################################################################################################################################################################

classdef running_median_object < handle 
    properties
        
        %Supplied Properties at initialization:
        number_of_frequencies_to_track
        number_of_frames_to_track
        
        %Calculated parameters:
        number_of_frames_to_keep_in_memory
        median_history_buffer_size
        running_phase_spectrum_matrix
        median_vec
        Next_pointer_chain
        Previous_pointer_chain;
        median_pointer
        current_buffer_index
        noise_gate_threshold_vec
        running_min_vec
        running_max_vec
    end
    
    properties(SetAccess = private)
        
    end

    
    methods
        
        %CONSTRUCTOR:
        function [running_signal_median_object] = running_median_object(number_of_frequencies_to_track,number_of_frames_to_track)
            
            %Insert input variables to object:
            running_signal_median_object.number_of_frequencies_to_track = number_of_frequencies_to_track;
            running_signal_median_object.number_of_frames_to_track = number_of_frames_to_track;
            
            running_signal_median_object.median_history_buffer_size = floor(running_signal_median_object.number_of_frames_to_track/2)*2+1;
            running_signal_median_object.number_of_frames_to_keep_in_memory = running_signal_median_object.median_history_buffer_size + 1;
            running_signal_median_object.median_vec = zeros(running_signal_median_object.number_of_frequencies_to_track,1);
            running_signal_median_object.noise_gate_threshold_vec = zeros(running_signal_median_object.number_of_frequencies_to_track,1);
            running_signal_median_object.running_min_vec = zeros(running_signal_median_object.number_of_frequencies_to_track,1);
            running_signal_median_object.running_max_vec = zeros(running_signal_median_object.number_of_frequencies_to_track,1);
            
            %Initialize pointer chains and running phase spectrum matrix 
            running_signal_median_object.running_phase_spectrum_matrix = zeros(running_signal_median_object.number_of_frequencies_to_track,running_signal_median_object.number_of_frames_to_keep_in_memory);
            running_signal_median_object.Next_pointer_chain = repmat([2:running_signal_median_object.median_history_buffer_size,running_signal_median_object.median_history_buffer_size+1,1],running_signal_median_object.number_of_frequencies_to_track,1);
            running_signal_median_object.Previous_pointer_chain = repmat([running_signal_median_object.median_history_buffer_size+1,1:running_signal_median_object.median_history_buffer_size],running_signal_median_object.number_of_frequencies_to_track,1);
            running_signal_median_object.median_pointer = ones(running_signal_median_object.number_of_frequencies_to_track,1)*(running_signal_median_object.median_history_buffer_size+1)/2;
            
            %Initialize running buffer to show where to put current spectrum:
            running_signal_median_object.current_buffer_index = 1;
            
            %Initialize internal parameters and variables:
%             reset(running_signal_median_object);
        end
        
        function reset(running_signal_median_object)
            %Calculate needed history buffer size and number of frames to keep in memory:
            running_signal_median_object.median_history_buffer_size = floor(running_signal_median_object.number_of_frames_to_track/2)*2+1;
            running_signal_median_object.number_of_frames_to_keep_in_memory = running_signal_median_object.median_history_buffer_size + 1;
            running_signal_median_object.median_vec = zeros(size(running_signal_median_object.number_of_frequencies_to_track,1),1);
            running_signal_median_object.noise_gate_threshold_vec = zeros(size(running_signal_median_object.number_of_frequencies_to_track,1),1);
            
            %Initialize pointer chains and running phase spectrum matrix 
            running_signal_median_object.running_phase_spectrum_matrix = zeros(running_signal_median_object.number_of_frequencies_to_track,running_signal_median_object.number_of_frames_to_keep_in_memory);
            running_signal_median_object.Next_pointer_chain = repmat([2:running_signal_median_object.median_history_buffer_size,running_signal_median_object.median_history_buffer_size+1,1],running_signal_median_object.number_of_frequencies_to_track,1);
            running_signal_median_object.Previous_pointer_chain = repmat([running_signal_median_object.median_history_buffer_size+1,1:running_signal_median_object.median_history_buffer_size],running_signal_median_object.number_of_frequencies_to_track,1);
            running_signal_median_object.median_pointer = ones(running_signal_median_object.number_of_frequencies_to_track,1)*(running_signal_median_object.median_history_buffer_size+1)/2;
            
            %Initialize running buffer to show where to put current spectrum:
            running_signal_median_object.current_buffer_index = 1;
        end
        
        function [estimated_noise_gate,current_running_min_vec,current_running_max_vec] = update_median(running_signal_median_object,current_sub_frame_spectrum)
            
            %Insert current spectrum into running phase spectrum matrix:
            running_signal_median_object.running_phase_spectrum_matrix(:,running_signal_median_object.current_buffer_index) = current_sub_frame_spectrum;
            
            %Perform running median tracking:
            [running_signal_median_object.median_vec,running_signal_median_object.running_min_vec,running_signal_median_object.running_max_vec,running_signal_median_object.median_pointer,running_signal_median_object.running_phase_spectrum_matrix,running_signal_median_object.Next_pointer_chain,running_signal_median_object.Previous_pointer_chain] = ...
                running_median_calculation(current_sub_frame_spectrum,...
                running_signal_median_object.running_phase_spectrum_matrix,...
                running_signal_median_object.Next_pointer_chain,...
                running_signal_median_object.Previous_pointer_chain,...
                running_signal_median_object.current_buffer_index,...
                running_signal_median_object.median_pointer,...
                running_signal_median_object.median_vec);                     
            
            current_running_min_vec = running_signal_median_object.running_min_vec;
            current_running_max_vec = running_signal_median_object.running_max_vec;
            
            %Update current buffer index (circular):
            running_signal_median_object.current_buffer_index = mod(running_signal_median_object.current_buffer_index,running_signal_median_object.median_history_buffer_size) + 1;
            
            %Smooth median over time to get noise estimation:
            estimated_noise_gate = 0.95*running_signal_median_object.noise_gate_threshold_vec + 0.05*running_signal_median_object.median_vec;
            running_signal_median_object.noise_gate_threshold_vec = estimated_noise_gate;
            
                  
       
        end
        
   
            
    end 
end












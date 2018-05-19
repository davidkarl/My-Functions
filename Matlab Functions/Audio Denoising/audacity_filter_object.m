classdef audacity_filter_object < handle 
    properties
        
        %Supplied Properties at initialization:
        Fs
        non_overlapping_samples_per_frame
        FFT_size
        
        %Possible Supplied Properties:
        noise_attenuation_in_dB
        sensitivity_in_dB
        frequency_smoothing_BW_in_Hz
        attack_time
        release_time
        spectral_activity_search_time
        noise_sensitivity_factor
        noise_attenuation_factor
        frequency_index_to_cutoff_algorithm_memory
        
        %Calculated Parameters:
        number_of_frequency_smoothing_bins
        attack_decay_number_of_samples
        attack_decay_number_of_blocks
        attack_decay_per_block
        release_decay_number_of_samples
        release_decay_number_of_blocks
        release_decay_per_block
        spectral_minimum_search_number_of_blocks
        gains_history_number_of_blocks
        center_of_search_window_in_gains_history
        start_of_search_window_in_gains_history
        stop_of_search_window_in_gains_history
        gains_history_full_flipped_indices
        current_search_window_indices_within_gains_history
        current_fft_update_index
        current_final_fft_index
        center_of_search_window_index
        
        %Initialized matrices:
        noise_attenuation_constant_vec
        noise_attenuation_constant_vec_full
    end
    
    properties(SetAccess = private)
        spectrums_history
        ffts_history
        gains_history
    end
    
    
    methods
        
        %CONSTRUCTOR (gives default values):
        function [audacity_filter_object] = audacity_filter_object(non_overlapping_samples_per_frame,Fs,FFT_size,frequency_index_to_cutoff_algorithm_memory)
           reset(audacity_filter_object,non_overlapping_samples_per_frame,Fs,FFT_size,frequency_index_to_cutoff_algorithm_memory);
        end
        
        %TOTAL Reset function (also sets parameters to default values):
        function reset(audacity_filter_object,non_overlapping_samples_per_frame,Fs,FFT_size,frequency_index_to_cutoff_algorithm_memory)
            
            %Insert input values to object:
            audacity_filter_object.non_overlapping_samples_per_frame = non_overlapping_samples_per_frame;
            audacity_filter_object.Fs = Fs;
            audacity_filter_object.FFT_size = FFT_size;
            audacity_filter_object.frequency_index_to_cutoff_algorithm_memory = frequency_index_to_cutoff_algorithm_memory;
            
            %Default Noise Reduction Parameters:
            audacity_filter_object.noise_attenuation_in_dB = -20;
            audacity_filter_object.sensitivity_in_dB = 6;
            audacity_filter_object.frequency_smoothing_BW_in_Hz = 0;
            audacity_filter_object.number_of_frequency_smoothing_bins = round((audacity_filter_object.frequency_smoothing_BW_in_Hz/Fs)*FFT_size);
            audacity_filter_object.attack_time = 0.1;
            audacity_filter_object.release_time = 0.1;
            audacity_filter_object.spectral_activity_search_time = 0.05;
            audacity_filter_object.noise_sensitivity_factor = 10^(audacity_filter_object.sensitivity_in_dB/10);
            audacity_filter_object.noise_attenuation_factor = 10^(audacity_filter_object.noise_attenuation_in_dB/20);
            
            
            %Initialize noise reduction matrices:
            reset_noise_reduction_matrices(audacity_filter_object);
                
        end
        
        function reset_noise_reduction_matrices(audacity_filter_object)
             %Noise reduction Parameters:
             audacity_filter_object.number_of_frequency_smoothing_bins = int8(round((audacity_filter_object.frequency_smoothing_BW_in_Hz/(audacity_filter_object.Fs/2)) * audacity_filter_object.non_overlapping_samples_per_frame));
             audacity_filter_object.attack_decay_number_of_samples = audacity_filter_object.attack_time*audacity_filter_object.Fs;
             audacity_filter_object.attack_decay_number_of_blocks = int8(1 + round( audacity_filter_object.attack_decay_number_of_samples / audacity_filter_object.non_overlapping_samples_per_frame ));
             audacity_filter_object.attack_decay_per_block = 10^( audacity_filter_object.noise_attenuation_in_dB/(20*double(audacity_filter_object.attack_decay_number_of_blocks)) );
             audacity_filter_object.release_decay_number_of_samples = audacity_filter_object.release_time*audacity_filter_object.Fs;
             audacity_filter_object.release_decay_number_of_blocks = int8(1 + round( audacity_filter_object.release_decay_number_of_samples / audacity_filter_object.non_overlapping_samples_per_frame ));
             audacity_filter_object.release_decay_per_block = 10^( audacity_filter_object.noise_attenuation_in_dB/(20*double(audacity_filter_object.release_decay_number_of_blocks)) );
             audacity_filter_object.spectral_minimum_search_number_of_blocks = int8(round( audacity_filter_object.spectral_activity_search_time * audacity_filter_object.Fs / audacity_filter_object.non_overlapping_samples_per_frame ));
             audacity_filter_object.spectral_minimum_search_number_of_blocks = int8(max( 2 , audacity_filter_object.spectral_minimum_search_number_of_blocks ));
             audacity_filter_object.gains_history_number_of_blocks = audacity_filter_object.attack_decay_number_of_blocks + audacity_filter_object.release_decay_number_of_blocks + 1;
             audacity_filter_object.gains_history_number_of_blocks = int8(max(audacity_filter_object.gains_history_number_of_blocks,audacity_filter_object.spectral_minimum_search_number_of_blocks));
             
             %Search window history indices:
             audacity_filter_object.center_of_search_window_in_gains_history = int8(1 + audacity_filter_object.release_decay_number_of_blocks);
             audacity_filter_object.start_of_search_window_in_gains_history = int8(audacity_filter_object.center_of_search_window_in_gains_history - floor(audacity_filter_object.spectral_minimum_search_number_of_blocks/2));
             audacity_filter_object.stop_of_search_window_in_gains_history = int8(audacity_filter_object.center_of_search_window_in_gains_history + ceil(audacity_filter_object.spectral_minimum_search_number_of_blocks/2));
             
             %Initialize audacity algorithm running indices and indices vecs:
             audacity_filter_object.gains_history_full_flipped_indices = int8(flip(1:1:audacity_filter_object.gains_history_number_of_blocks)');
             audacity_filter_object.current_search_window_indices_within_gains_history = int8(audacity_filter_object.gains_history_full_flipped_indices(audacity_filter_object.start_of_search_window_in_gains_history:audacity_filter_object.stop_of_search_window_in_gains_history));
             audacity_filter_object.current_fft_update_index = int8(audacity_filter_object.gains_history_full_flipped_indices(end));
             audacity_filter_object.current_final_fft_index = int8(audacity_filter_object.gains_history_full_flipped_indices(end-1));
             audacity_filter_object.center_of_search_window_index = int8(audacity_filter_object.gains_history_full_flipped_indices(audacity_filter_object.center_of_search_window_in_gains_history));
             
             %Initialize Matrices containing spectrums and gains history:
             audacity_filter_object.spectrums_history = zeros(audacity_filter_object.frequency_index_to_cutoff_algorithm_memory,audacity_filter_object.gains_history_number_of_blocks);
             audacity_filter_object.ffts_history = zeros(audacity_filter_object.FFT_size,audacity_filter_object.gains_history_number_of_blocks);
             audacity_filter_object.gains_history = ones(audacity_filter_object.frequency_index_to_cutoff_algorithm_memory,audacity_filter_object.gains_history_number_of_blocks) * audacity_filter_object.noise_attenuation_factor;
             audacity_filter_object.noise_attenuation_constant_vec = audacity_filter_object.noise_attenuation_factor*ones(audacity_filter_object.frequency_index_to_cutoff_algorithm_memory,1);
             audacity_filter_object.noise_attenuation_constant_vec_full = audacity_filter_object.noise_attenuation_factor*ones(audacity_filter_object.FFT_size,1);
        end
        
 
                 
        function [fft_after_gain_function,final_gain_function] = filter(audacity_signal_filter_object,current_sub_frame_fft,current_sub_frame_spectrum,noise_gate_threshold_vec)
            
            %Update histories with current fft, spectrum and gain factor:
%             tic
            audacity_signal_filter_object.ffts_history(:,audacity_signal_filter_object.current_fft_update_index) = current_sub_frame_fft;
            audacity_signal_filter_object.spectrums_history(:,audacity_signal_filter_object.current_fft_update_index) = current_sub_frame_spectrum;
            audacity_signal_filter_object.gains_history(:,audacity_signal_filter_object.current_fft_update_index) = audacity_signal_filter_object.noise_attenuation_factor;
                            
            
            %Find Indices above noise:
            noise_classification_flag_counter=1;
            search_window = audacity_signal_filter_object.spectrums_history(:,audacity_signal_filter_object.current_search_window_indices_within_gains_history);
            if noise_classification_flag_counter==1
                %check if second greatest in search window is above noise threshold
                current_sorted_search_window_spectrums = sort(search_window,2);
                indices_above_noise_gate = uint16(find(current_sorted_search_window_spectrums(:,end-1) > audacity_signal_filter_object.noise_sensitivity_factor * noise_gate_threshold_vec));
            elseif noise_classification_flag_counter==2
                %check if current window mean is above noise threshold
                current_mean_search_window_spectrums = mean(search_window,2);
                indices_above_noise_gate = uint16(find(current_mean_search_window_spectrums > noise_sensitivity_factor1 * noise_gate_threshold_vec));
            end
            
            
            %DECAY THE GAIN IN BOTH DIRECTIONS:
            gains_history_above_noise = audacity_signal_filter_object.gains_history(indices_above_noise_gate,:);
            if ~isempty(indices_above_noise_gate)
                gains_history_above_noise(:,audacity_signal_filter_object.center_of_search_window_index) = 1;
                noise_attenuation_constants = audacity_signal_filter_object.noise_attenuation_factor * ones(length(indices_above_noise_gate),1);
                %(*) if the gain before or after is lower then what one gets when we decay the current center gain
                % then RAISE it according to decay rate from center gain:
                %HOLD (backward in time):
                for history_frame_counter = audacity_signal_filter_object.center_of_search_window_in_gains_history-1 : -1 : 1
                    current_index = audacity_signal_filter_object.gains_history_full_flipped_indices(history_frame_counter+1);
                    next_index = audacity_signal_filter_object.gains_history_full_flipped_indices(history_frame_counter);
                    gains_history_above_noise(:,next_index) = max( gains_history_above_noise(:,current_index)*audacity_signal_filter_object.attack_decay_per_block, max(noise_attenuation_constants,gains_history_above_noise(:,next_index)) );
                end
                %RELEASE (forward in time):
                for history_frame_counter = audacity_signal_filter_object.center_of_search_window_in_gains_history+1 : +1 : audacity_signal_filter_object.gains_history_number_of_blocks
                    current_index = audacity_signal_filter_object.gains_history_full_flipped_indices(history_frame_counter-1);
                    next_index = audacity_signal_filter_object.gains_history_full_flipped_indices(history_frame_counter);
                    gains_history_above_noise(:,next_index) = max( gains_history_above_noise(:,current_index)*audacity_signal_filter_object.release_decay_per_block, max(noise_attenuation_constants,gains_history_above_noise(:,next_index)) );
                end
            end
            
            %Get current end fft and gain:
            audacity_signal_filter_object.gains_history(indices_above_noise_gate,:) = gains_history_above_noise;
            final_gain_function = audacity_signal_filter_object.noise_attenuation_factor*ones(audacity_signal_filter_object.FFT_size,1);
            final_gain_function(1:audacity_signal_filter_object.frequency_index_to_cutoff_algorithm_memory) = audacity_signal_filter_object.gains_history(:,audacity_signal_filter_object.current_final_fft_index);
            final_gain_function(end/2+1:end/2+1+audacity_signal_filter_object.frequency_index_to_cutoff_algorithm_memory-1) = audacity_signal_filter_object.gains_history(:,audacity_signal_filter_object.current_final_fft_index);
            
            %Multiply current fft by gain function:
            fft_after_gain_function = audacity_signal_filter_object.ffts_history(:,audacity_signal_filter_object.current_final_fft_index).*final_gain_function + eps*1i;
            
            %Increment indices in gains history indices:
            audacity_signal_filter_object.gains_history_full_flipped_indices = mod(audacity_signal_filter_object.gains_history_full_flipped_indices,audacity_signal_filter_object.gains_history_number_of_blocks)+1;
            audacity_signal_filter_object.current_fft_update_index = audacity_signal_filter_object.gains_history_full_flipped_indices(end);
            audacity_signal_filter_object.current_final_fft_index = audacity_signal_filter_object.gains_history_full_flipped_indices(end-1);
            audacity_signal_filter_object.center_of_search_window_index = audacity_signal_filter_object.gains_history_full_flipped_indices(audacity_signal_filter_object.center_of_search_window_in_gains_history);
            audacity_signal_filter_object.current_search_window_indices_within_gains_history = audacity_signal_filter_object.gains_history_full_flipped_indices(audacity_signal_filter_object.start_of_search_window_in_gains_history:audacity_signal_filter_object.stop_of_search_window_in_gains_history);
%             toc
            
        end  
          
        
        
        
        
    end %END OF METHODS
end







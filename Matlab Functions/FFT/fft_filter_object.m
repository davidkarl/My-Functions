classdef fft_filter_object < handle 
    properties
        %Possible Supplied Properties:
        samples_per_frame
        overlap_samples_per_frame
        flag_specify_overlap_or_use_minimum_needed
        filter_type_low_high_band_free_style_equalizer
        Fs
        start_frequency
        stop_frequency
        filter_length
        window_type
        kaiser_parameter
        free_style_frequencies
        free_style_values
        equalizer_max_frequency
        equalizer_number_of_frequency_bands
        
        %Calculated Parameters:
        signal_filter
        signal_filter_fft
        filter_group_delay
        filter_spectrum
        FFT_size
        frame_window
        
        %Later Initialized:
        previous_raw_frame_overlapping_samples_for_buffering
        previous_filtered_frame_edge_for_overlap_add
    end
    properties(SetAccess = private)
        tail_index
        ndx_SPEC
    end

    
    methods
        
        
        function filter_object = fft_filter_object
            %do nothing
        end
        
        function add_filter(filter_object)
            
            
        end
        
        
        function create_filter(filter_object)
            
            %get standard filter parameters:
            filter_window_name_temp = filter_object.window_type;
            start_frequency_temp = filter_object.start_frequency;
            stop_frequency_temp = filter_object.stop_frequency;
            filter_length_temp = filter_object.filter_length;
            filter_order_temp = filter_length_temp - 1;
            kaiser_parameter_temp = filter_object.kaiser_parameter;
            Fs_temp = filter_object.Fs;
            
            %get free style filter parameters:
            filter_type_low_high_band_free_style_equalizer_equalizer_temp = filter_object.filter_type_low_high_band_free_style_equalizer;
            free_style_frequencies_temp = filter_object.free_style_frequencies;
            free_style_amplitude_gains_dB_temp = filter_object.free_style_values;
            
            %get equalizer filter parameters:
            equalizer_number_of_frequency_bands_temp = filter_object.equalizer_number_of_frequency_bands;
            equalizer_max_frequency_temp = filter_object.equalizer_max_frequency;
            
            %actually create the filter and get group delay:
            if strcmp(filter_type_low_high_band_free_style_equalizer_equalizer_temp,'free_style')
                [signal_filter_temp] = get_filter_1D_arbitrary(filter_order_temp,Fs_temp,free_style_frequencies_temp,free_style_amplitude_gains_dB_temp,filter_window_name_temp);
            elseif strcmp(filter_type_low_high_band_free_style_equalizer_equalizer_temp,'equalizer')
                [signal_filter_temp] = get_filter_1D_equalizer(filter_order_temp,Fs_temp,equalizer_number_of_frequency_bands_temp,equalizer_max_frequency_temp,free_style_amplitude_gains_dB_temp,filter_window_name_temp);
            else
                %standard: "lowpass","highpass','bandpass','bandstop'
                [signal_filter_temp] = get_filter_1D(filter_window_name_temp,kaiser_parameter_temp,filter_length_temp,Fs_temp,start_frequency_temp,stop_frequency_temp,filter_type_low_high_band_free_style_equalizer_equalizer_temp);
            end
            filter_group_delay_temp = round(mean(grpdelay(signal_filter_temp)));
            
                
            %get overlap samples per frame:
            samples_per_frame_temp = filter_object.samples_per_frame;
            if filter_object.flag_specify_overlap_or_use_minimum_needed==1
                overlap_samples_per_frame_temp = filter_object.overlap_samples_per_frame; 
            else
                overlap_samples_per_frame_temp = filter_order_temp;
            end
            
            %create frame window (different from fir2 window to create impulse):
            frame_overlap_edge_window = make_column(hanning(2*overlap_samples_per_frame_temp-1,'periodic'));
            frame_window_temp = [frame_overlap_edge_window(1:overlap_samples_per_frame_temp);ones(samples_per_frame_temp-2*overlap_samples_per_frame_temp,1);frame_overlap_edge_window(overlap_samples_per_frame_temp:end)];
                        
            %get filter FFT_size, fft, and normalized frame window while you're at it:
            FFT_size_temp = 2^nextpow2(samples_per_frame_temp + filter_length_temp - 1);
            frame_window_temp = frame_window_temp ./ (sum(frame_window_temp*2) * FFT_size_temp;
            signal_filter_fft_temp = make_column(fft(signal_filter_temp.Numerator,FFT_size_temp));
            
            %Initialize overlap parts:
            previous_raw_frame_overlapping_samples_for_buffering_temp = zeros(overlap_samples_per_frame_temp,1);
            previous_filtered_frame_for_overlap_add_temp = zeros(overlap_samples_per_frame_temp,1);

            %assign things to filter_object:
            filter_object.filter_group_delay = filter_group_delay_temp;
            filter_object.signal_filter_fft = signal_filter_fft_temp;
            filter_object.filter_spectrum = abs(fftshift(signal_filter_fft_temp)).^2;
            filter_object.signal_filter = signal_filter_temp;
            filter_object.overlap_samples_per_frame = overlap_samples_per_frame_temp;
            filter_object.FFT_size = FFT_size_temp;
            filter_object.frame_window = frame_window_temp;
            filter_object.previous_raw_frame_overlapping_samples_for_buffering = previous_raw_frame_overlapping_samples_for_buffering_temp;
            filter_object.previous_filtered_frame_edge_for_overlap_add = previous_filtered_frame_for_overlap_add_temp;
        end
        
        
         
        function [filtered_signal_final] = filter(filter_object,input_signal,flag_time_domain_or_fft,flag_buffered_and_windowed_or_not)
            
            %get filter parameters:
            samples_per_frame_temp = filter_object.samples_per_frame;
            overlap_samples_per_frame_temp = filter_object.overlap_samples_per_frame;
            previous_raw_frame_overlapping_samples_for_buffering_temp = filter_object.previous_raw_frame_overlapping_samples_for_buffering;
            frame_window_temp = filter_object.frame_window;
            FFT_size_temp = filter_object.FFT_size;
            signal_filter_fft_temp = filter_object.signal_filter_fft;
            filter_group_delay_temp = filter_object.filter_group_delay;
            previous_filtered_frame_edge_for_overlap_add_temp = filter_object.previous_filtered_frame_edge_for_overlap_add;
            
            %buffer unbuffered signal if needed:
            if flag_time_domain_or_fft==1 && flag_buffered_and_windowed_or_not==2
                %buffer signal:
                input_signal_cut = input_signal(1:end-overlap_samples_per_frame_temp);
                buffered_and_windowed_signal = [previous_raw_frame_overlapping_samples_for_buffering_temp ; input_signal_cut];
                previous_raw_frame_overlapping_samples_for_buffering_temp = input_signal(end-overlap_samples_per_frame_temp+1:end);
                
                %window overlapping samples at frame edges:
                buffered_and_windowed_signal = buffered_and_windowed_signal .* frame_window_temp;
            else
                %input signal is already buffered and windowed:
                buffered_and_windowed_signal = input_signal;
            end
            
            %calculate filtered signal in fft domain:
            if flag_time_domain_or_fft==1 
                %TIME DOMAIN:
                
                %Calculate buffered and windowed frame FFT:
                input_signal_fft = fft(buffered_and_windowed_signal, FFT_size_temp);
                
                %Calculate time domain filtered signal:
                filtered_signal = real(ifft(input_signal_fft .* signal_filter_fft_temp));
                
            else
                %FFT DOMAIN:
                filtered_signal = real(ifft(input_signal .* signal_filter_fft_temp));
            end
            
            
            %get valid part of time domain filtered signal:
            filtered_signal_valid = filtered_signal(filter_group_delay_temp+1:filter_group_delay_temp+samples_per_frame_temp);
            
            %overlap-add:
            filtered_signal_final = filtered_signal_valid(1:overlap_samples_per_frame_temp) ...
                 + previous_filtered_frame_edge_for_overlap_add_temp;
            previous_filtered_frame_edge_for_overlap_add_temp = filtered_signal_valid(end-overlap_samples_per_frame_temp+1:end);
            
            
            %Update filter object:
            filter_object.previous_raw_frame_overlapping_samples_for_buffering = previous_raw_frame_overlapping_samples_for_buffering_temp;
            filter_object.previous_filtered_frame_edge_for_overlap_add = previous_filtered_frame_edge_for_overlap_add_temp;            
           
        end
                
        
    end
end







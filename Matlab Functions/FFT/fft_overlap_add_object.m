classdef fft_overlap_add_object < handle 
    properties
        %Possible Supplied Properties:
        Fs
        samples_per_frame
        overlap_samples_per_frame
        signal_filter
        flag_already_windowed_or_not
        number_of_samples_to_skip_at_ifft_output
                
        %Calculated Parameters:
        non_overlapping_samples_per_frame
        filter_fft
        group_delay
        filter_length
        filter_spectrum
        FFT_size
        frame_window
        impulse_response
        
        %Initialized:
        lookahead_buffer_for_overlap_add
        
    end
    
    properties(SetAccess = private)
        
    end

    
    methods
        
        %CONSTRUCTOR:
        function [filter_object] = fft_overlap_add_object(signal_filter,Fs,samples_per_frame,overlap_samples_per_frame,flag_already_windowed_or_not,number_of_samples_to_skip_at_ifft_output)
            
            %get basic filtering and framing_parameters:
            filter_object.samples_per_frame = samples_per_frame;
            filter_object.overlap_samples_per_frame = overlap_samples_per_frame;
            filter_object.non_overlapping_samples_per_frame = samples_per_frame-overlap_samples_per_frame;
            filter_object.Fs = Fs;
            filter_object.signal_filter = signal_filter;
            filter_object.filter_length = length(signal_filter.Numerator);
            filter_object.impulse_response = signal_filter.impz;
            filter_object.group_delay = floor(filter_object.filter_length/2) + 1; %FIR
            filter_object.number_of_samples_to_skip_at_ifft_output = number_of_samples_to_skip_at_ifft_output;
            
            %if the frames aren't already windowed then create window for them:
            filter_object.flag_already_windowed_or_not = flag_already_windowed_or_not;
            if flag_already_windowed_or_not==1
                [filter_object.frame_window] = get_approximately_COLA_window(samples_per_frame,overlap_samples_per_frame);
            else
                filter_object.frame_window = ones(samples_per_frame,1);
            end 
             
            %get filter FFT_size and FFT::
            filter_object.FFT_size = 2^nextpow2(filter_object.samples_per_frame + filter_object.filter_length - 1);
            filter_object.filter_fft = make_column(fft(filter_object.signal_filter.Numerator,filter_object.FFT_size));
            filter_object.filter_spectrum = abs(fftshift(filter_object.filter_fft)).^2;
            
            %Initialize lookahead buffer for overlap add operation:
            filter_object.lookahead_buffer_for_overlap_add = zeros(filter_object.FFT_size,1);
        end
         
        
        function cascade(filter_object,new_filter)
            
            %i want to add another filter after currently defined one so i need to merge them:
            cascaded_filter = dfilt.cascade(filter_object.signal_filter , new_filter);
            cascaded_filter = dfilt.dffir(cascaded_filter.impz);
            cascaded_filter.PersistentMemory = true;
            cascaded_filter.States = 0;
            
            %updatae filter object:
            filter_object.filter_length = length(cascaded_filter.impz);
            filter_object.impulse_response = cascaded_filter.impz;
            filter_object.group_delay = floor(filter_object.filter_length/2) + 1;
            filter_object.FFT_size = 2^nextpow2(filter_object.samples_per_frame + filter_object.filter_length - 1);
            filter_object.filter_fft = fft(cascaded_filter.impz , filter_object.FFT_size);
            filter_object.filter_spectrum = abs(fftshift(filter_object.filter_fft)).^2;
            filter_object.signal_filter = cascaded_filter;
            
        end
        
        function equate_latency(filter_object,filter_object2)
           %the purpose here is to equal current filter's latency with the latency of a bigger second filter
           %such that they come with the same delay and can be use at the same time to add/substract/do something
           %to both of them at the same time (like anaytic signal):
           %another way i think this can be done is by increasing the overlap between frames (probably i'm wrong):
           
           %calculate latency/zero padding needed:
           if isa(filter_object2,'fft_overlap_add_object')
               filter_latency_to_add = floor(filter_object2.filter_length/2) - floor(filter_object.filter_length/2);
           elseif isa(filter_object2,'dfilt.dffir')
               filter_latency_to_add = floor(filter_object2.impzlength/2) - floor(filter_object.impzlength/2);
           end
           
           %check which filter to change:
           if filter_latency_to_add > 0
              %second filter is larger: 
              if isa(filter_object2,'fft_overlap_add_object')
                  smaller_impz = filter_object.signal_filter.Numerator;
              elseif isa(filter_object2,'dfilt.dffir')
                  smaller_impz = filter_object.Numerator;
              end
              smaller_impz = smaller_impz(:);
              smaller_impz = [zeros(abs(filter_latency_to_add),1) ; smaller_impz];
              smaller_impz = make_column(smaller_impz);
               
              %create new filter:
              new_filter = dfilt.dffir(smaller_impz);
              
              %update filter object:
              filter_object.filter_length = length(smaller_impz);
              filter_object.impulse_response = make_column(smaller_impz);
              filter_object.group_delay = floor(filter_object.filter_length/2) + 1;
              filter_object.FFT_size = 2^nextpow2(filter_object.samples_per_frame + filter_object.filter_length - 1);
              filter_object.filter_fft = fft(smaller_impz , filter_object.FFT_size);
              filter_object.filter_spectrum = abs(fftshift(filter_object.filter_fft)).^2;
              filter_object.signal_filter = new_filter;
           
           else
              %first filter is larger:
              if isa(filter_object2,'fft_overlap_add_object')
                  smaller_impz = filter_object2.signal_filter.Numerator;
              elseif isa(filter_object2,'dfilt.dffir')
                  smaller_impz = filter_object2.Numerator;
              end
              smaller_impz = smaller_impz(:);
              smaller_impz = [zeros(abs(filter_latency_to_add),1) ; smaller_impz];
              smaller_impz = make_column(smaller_impz);
              
              %create new filter:
              new_filter = dfilt.dffir(smaller_impz);
              
              %update filter object:
              filter_object2.filter_length = length(smaller_impz);
              filter_object2.impulse_response = make_column(smaller_impz);
              filter_object2.group_delay = floor(filter_object2.filter_length/2) + 1;
              filter_object2.FFT_size = 2^nextpow2(filter_object2.samples_per_frame + filter_object2.filter_length - 1);
              filter_object2.filter_fft = fft(smaller_impz , filter_object2.FFT_size);
              filter_object2.filter_spectrum = abs(fftshift(filter_object2.filter_fft)).^2;
              filter_object2.signal_filter = new_filter;
              
           end        
           
        end
        
        function add_latency(filter_object,samples_or_object)
            
            if isa(samples_or_object,'fft_overlap_add_object') 
                samples_to_add = floor(samples_or_object.filter_length/2)+1;
            elseif isa(samples_or_object,'dfilt.dffir')
                samples_to_add = floor(length(samples_or_object.Numerator)/2)+1;
            elseif isnumeric(samples_or_object)
                samples_to_add = samples_or_object;
            end
            smaller_impz = filter_object.impulse_response;
            smaller_impz = smaller_impz(:);
            smaller_impz = [zeros(samples_to_add,1) ; smaller_impz];
            smaller_impz = make_column(smaller_impz);
            
            %create new filter:
            new_filter = dfilt.dffir(smaller_impz);
            
            %update filter object:
            filter_object.filter_length = length(smaller_impz);
            filter_object.impulse_response = make_column(smaller_impz);
            filter_object.group_delay = floor(filter_object.filter_length/2) + 1;
            filter_object.FFT_size = 2^nextpow2(filter_object.samples_per_frame + filter_object.filter_length - 1);
            filter_object.filter_fft = fft(smaller_impz , filter_object.FFT_size);
            filter_object.filter_spectrum = abs(fftshift(filter_object.filter_fft)).^2;
            filter_object.signal_filter = new_filter;
            
            
        end
         
        function [filtered_signal_final_valid] = filter(filter_object,input_signal,flag_time_domain_or_fft)
            
            %window signals if needed:
            if flag_time_domain_or_fft==1 && filter_object.flag_already_windowed_or_not==2
                input_signal = input_signal .* filter_object.frame_window;
            end
            
            %calculate filtered signal in fft domain:
            if flag_time_domain_or_fft==1 
                %TIME DOMAIN:
                
                %Calculate buffered and windowed frame FFT:
                input_signal_fft = fft(input_signal, filter_object.FFT_size);
                
                %Calculate time domain filtered signal:
                filtered_signal = real(ifft(input_signal_fft .* filter_object.filter_fft));
                
            else
                %FFT DOMAIN:
                filtered_signal = real(ifft(input_signal .* filter_object.filter_fft));
            end
            
            %overlap-add:
%             filter_object.lookahead_buffer_for_overlap_add(1:end-filter_object.number_of_samples_to_skip_at_ifft_output) = ...
%                 filter_object.lookahead_buffer_for_overlap_add(1:end-filter_object.number_of_samples_to_skip_at_ifft_output) + filtered_signal(filter_object.number_of_samples_to_skip_at_ifft_output+1:end);
            filter_object.lookahead_buffer_for_overlap_add = ...
                filter_object.lookahead_buffer_for_overlap_add + filtered_signal;
            
            %get current valid part of overlap-add:
            filtered_signal_final_valid = filter_object.lookahead_buffer_for_overlap_add(1:filter_object.non_overlapping_samples_per_frame);
            
            %shift lookahead buffer tail to beginning of buffer for next overlap-add:
            filter_object.lookahead_buffer_for_overlap_add = [filter_object.lookahead_buffer_for_overlap_add(filter_object.non_overlapping_samples_per_frame+1:end) ; zeros(filter_object.non_overlapping_samples_per_frame,1)];
            
        end  
                
        
    end
end







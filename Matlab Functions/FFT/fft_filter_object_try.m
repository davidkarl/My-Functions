classdef fft_filter_object_try < handle 
    properties
        NFFT=4096
        Fs=44100
        M_tail_length=257
        batch_sz=500
        % This is an example input from the GUI you will do
        freq_vals=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1550,1600, 1650, 2000, 2500, 3000]
        gain_vals=[-1, -1, 0  ,  0 , 0.3, 0  , 0  , 0  , 0  , 0  , 0   ,  -1 ,  0  ,  0  ,-1  , 0   , 0   , 0   ,   -1]*80% dB
        send_data
        %SPEC
        out
        L  %
        FIR
        window
        overlap_samples_per_frame
    end
    properties(SetAccess = private)
        tail_index
        ndx_SPEC
%         min_index
    end
    %     properties (Dependent)
    %         L  %
    %         FIR
    %         window
    %         overlap
    %     end
    
    methods
        
        function obj=fft_filter_object
            reset(obj);
        end
        
        
        
        function reset(obj)
            %set valid data length:
            obj.L = obj.NFFT - obj.M_tail_length + 1;
            
            %set overlap percentage to 50 (maybe change later!):
            obj.overlap_samples_per_frame = obj.L/2;
            
            %set nonoverlapping length:
            non_overlapping_samples_per_frame = obj.L - obj.overlap_samples_per_frame;
            
            %initial tail index:
            obj.tail_index = 1;%obj.batch_sz-fix((obj.batch_sz-obj.overlap)/(inc))*inc;
            
            %set max length of data to filter:
            nmax = obj.batch_sz + obj.L - 1; 
            
            %initialize input data:
            obj.send_data.x = zeros(nmax,1);
            
            %number of batches to divide maximum data into
            ncol = fix((nmax-obj.overlap_samples_per_frame)/(non_overlapping_samples_per_frame));
            
            %time domain buffer max:
            obj.out.x = zeros(ncol * non_overlapping_samples_per_frame + obj.NFFT,1);
            
            %set filter window to hamming (change to selectable):
            obj.window = hamming(obj.L,'periodic');
            
            %normalize filter window:
            obj.window = obj.window./(sum(obj.window)*2)*obj.NFFT;
            
            %create actual filter:
            make_FIR(obj);
        end
        
        
        function make_FIR(obj)
            %get fft center index:
            fft_center_index = (obj.NFFT/2);
            
            %change to more accurate scale:
            freq = linspace(0,obj.Fs/2,fft_center_index+1);
            
            
            half_fil_vals=interp1(obj.freq_vals,obj.gain_vals,freq);
            figure;plot(freq,half_fil_vals)
            full_fil_vals = [half_fil_vals, conj(half_fil_vals(end-1:-1:2))];
            full_fil_vals=10.^(full_fil_vals/20);
            td_filter=real(fft(full_fil_vals));
            
            for i=0:(obj.M_tail_length-1)/2; %/2(i=0;i<=(M-1)/2;i++)
                %    {   //Windowing - could give a choice, fixed for now -
                % //      double mult=0.54-0.46*cos(2*M_PI*(i+(M-1)/2.0)/(M-1));   //Hamming
                % //Blackman
                mult=0.42-0.5*cos(2*pi*(i+(obj.M_tail_length-1)/2.0)/(obj.M_tail_length-1))+.08*cos(4*pi*(i+(obj.M_tail_length-1)/2.0)/(obj.M_tail_length-1));
                td_filter(i+1)=td_filter(i+1)*mult;
                if (i~=0)
                    td_filter(obj.NFFT+1-i)=td_filter(obj.NFFT+1-i)*mult;
                end
            end
            
            %padding
            for i = ((obj.M_tail_length-1)/2+1):fft_center_index 
                %for(;i<=mWindowSize/2;i++)
                td_filter(i)=0;
                td_filter(obj.NFFT+1-i)=0;
            end
            
            tempr=zeros(obj.M_tail_length,1);
            for i=1:(obj.M_tail_length-1)/2
                tempr((obj.M_tail_length-1)/2+i)=td_filter(i);
                tempr(i)=td_filter(obj.NFFT-(obj.M_tail_length-1)/2+i);
            end
            tempr(obj.M_tail_length)=td_filter((obj.M_tail_length-1)/2 +1);
            td_filter(1:obj.M_tail_length)=tempr;
            td_filter(obj.M_tail_length+1:end)=0;
            obj.FIR=(fft(td_filter)/length(td_filter))';
            hold all;plot(freq,log10(abs(obj.FIR(1:obj.NFFT/2+1)))*20);
        end
        
         
        function [valid_data,last_valid] = filter(obj,X)
            % send_data contains the tail of the previous frames as input, so
            % the new data is put after the last value of the tail
            obj.send_data.x(obj.tail_index : obj.tail_index+obj.batch_sz-1) = X;
            
            %update last index:
            last_index = obj.tail_index + obj.batch_sz - 1;
            
            %get non overlapping samples per frame:
            non_overlapping_samples_per_frame = obj.L - obj.overlap_samples_per_frame;
            
            %set nuber of batches to divide data into:
            number_of_batches = fix((last_index-obj.overlap_samples_per_frame)/(non_overlapping_samples_per_frame));
            
            %go over the different columns/batches and filter them:
            for batch_index=1:number_of_batches
                %get start and stop indices of curent batch within data:
                start_index=(batch_index-1)*non_overlapping_samples_per_frame+1;
                stop_index = start_index + obj.L - 1;
                
                %multiply current batch by filter window:
                data_after_window = obj.send_data.x(start_index:stop_index) .* obj.window;
                
                %fft data after window and multiply it by fft domain FIR:
                temp = fft(data_after_window,obj.NFFT).*obj.FIR;
                
                %use ifft to get back filtered signal:
                filtered_signal = real(ifft(temp,obj.NFFT,1));
                                
                %overlap add time signal
                end_index = start_index + obj.NFFT - 1;
                obj.out.x(start_index:end_index) = obj.out.x(start_index:end_index) + filtered_signal;
            end
             
            %copy the tail of the unprocessed to the first part send_data for the next time the
            %function is called
            last_used = number_of_batches * non_overlapping_samples_per_frame;%=(ncols-1)*inc+L-overlap
            tail_length = last_index - last_used;
            obj.send_data.x(1:tail_length) = obj.send_data.x(last_used+1:last_index);
            obj.tail_index = tail_length+1;
            
            %extract the valid data from out, and move the tail processed
            %data to the beginning of the out buffer. 
            last_valid = last_used;%-obj.overlap;
            valid_data.x = obj.out.x(1:last_valid);
            ol2 = obj.NFFT - non_overlapping_samples_per_frame;
            obj.out.x(1:ol2)=obj.out.x(last_valid+1:last_valid+ol2);
            obj.out.x(ol2+1:end)=0;
        end
                
        
    end
end







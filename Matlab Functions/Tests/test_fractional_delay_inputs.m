%I ASSUME THAT THE LENGTH OF THE SPEECH IS SMALLER THAN THE SIZE OF THE ANALYTIC SIGNAL RECORDED.
%I ASSUME THAT I CONVERTED THE SPEECH .WAV FILE TO .BIN WITH PROPER UPSAMPLING TO EQUAL ANALYTIC SIGNAL .BIN FILE

%GET ANALYTIC SIGNAL TO INSERT THE SIGNAL INTO ITS PHASE:
file_name='turbulence_1400pm_new';
directory='C:\Users\master\Desktop\matlab\sol turbulence experiment\sound bin files for speech enhancement purposes';
file_name_with_bin = strcat(file_name,'.bin');
full_file_name_turbulence = fullfile(directory,file_name_with_bin);

%GET SPEECH SIGNAL TO INSERT INTO THE ANALYTIC SIGNAL PHASE:
file_name='towel_speaking_audio';
directory='C:\Users\master\Desktop\matlab';
file_name_with_bin = strcat(file_name,'.bin');
full_file_name_speech = fullfile(directory,file_name_with_bin);

%close previous instances:
try
   fclose(fid_turbulence);
   fclose(fid_turbulence2);
   fclose(fid_speech);
   fclose('all');
   release(fractional_delay_object);
   release(audio_player_object);
   release(audio_file_writer_demodulated);
   release(analytic_signal_object);
   release('all');
catch
end 
 
%initialize analytic signal fid:
fid_turbulence = fopen(full_file_name_turbulence,'r');
fid_turbulence2 = fopen(full_file_name_turbulence,'r');
number_of_elements_in_file = fread(fid_turbulence,1,'double');
Fs = fread(fid_turbulence,1,'double');
down_sample_factor = round(Fs/44100);
Fs_downsampled = Fs/down_sample_factor;

%initialize speech signal fid:
fid_speech = fopen(full_file_name_speech,'r');
number_of_elements_in_file_speech = fread(fid_speech,1,'double');
Fs_speech = fread(fid_speech,1,'double');
down_sample_factor = round(Fs_speech/44100);
Fs_downsampled_speech = Fs_speech/down_sample_factor;
 
%Multi-Channel properties:
number_of_channels = 1;

%Decide on number of samples per frame:
time_for_FFT_frame = 0.1; %100[mSec]
counts_per_time_frame = round(Fs*time_for_FFT_frame/5)*5;
samples_per_frame=counts_per_time_frame;
number_of_frames = floor(number_of_elements_in_file_speech/samples_per_frame);

%decide on speech peak value:
speech_peak_value = 0.3;
counter=1;
peak_current_value = 0; 
while counter<number_of_frames-10
    current_frame = fread(fid_speech,samples_per_frame,'double');
    peak_current_value = max(peak_current_value,max(abs(current_frame)));
    counter=counter+1; 
end
speech_scaling = peak_current_value/speech_peak_value;
frewind(fid_speech);
bla = fread(fid_speech,2,'double');


%Default Initial Values:
flag_PM_or_FM=1;
if flag_PM_or_FM==1
   PM_FM_str='PM'; 
else
   PM_FM_str='FM';
end
    
%audio player object:
audio_player_object = dsp.AudioPlayer;
audio_player_object.SampleRate = 44100;
audio_player_object.QueueDuration = 7;

%analytic signal object:
analytic_signal_object = dsp.AnalyticSignal;
analytic_signal_object.FilterOrder = 100;

%basic filter parameters:
Fc = 12000; %initial Fc
BW = 6000; %initial BW;
carrier_filter_parameter=20;
signal_filter_parameter=20;
carrier_filter_length = 128*10;
signal_filter_length = 128*10;
%carrier filter:
filter_name_carrier = 'hann';
f_low_cutoff_carrier = Fc-BW/2;
f_high_cutoff_carrier = Fc+BW/2;
carrier_filter_type = 'bandpass';
[carrier_filter] = get_filter_1D(filter_name_carrier,carrier_filter_parameter,carrier_filter_length,Fs,f_low_cutoff_carrier,f_high_cutoff_carrier,carrier_filter_type);
carrier_filter_object = dsp.FIRFilter('Numerator',carrier_filter.Numerator);
%signal filter:
filter_name_signal = 'hann';
signal_filter_type = 'bandpass';
signal_start_frequency=200;
signal_stop_frequency=2000;
[signal_filter] = get_filter_1D(filter_name_signal,signal_filter_parameter,signal_filter_length,Fs,signal_start_frequency,signal_stop_frequency,signal_filter_type);
signal_filter_object = dsp.FIRFilter('Numerator',signal_filter.Numerator);    


%save demodulation to wav file if wanted:
flag_save_to_wav=0;
if flag_save_to_wav==1
    audio_file_writer_demodulated = dsp.AudioFileWriter;
    audio_file_writer_demodulated.Filename = strcat(fullfile(directory,file_name), PM_FM_str ,' final demodulated audio ', ' ', num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),', number of receivers = ',num2str(number_of_channels),', [Hz]','.wav');
    audio_file_writer_demodulated.SampleRate = 44100;
end

%fractional delay input:
fractional_delay_object = dsp.VariableFractionalDelay;

  
counter=1;
multiplication_factor=2;
delay_in_seconds = 10;
delay_in_samples = floor(delay_in_seconds*Fs);
fractional_delay_needed = delay_in_seconds*Fc - floor(delay_in_seconds*Fc); 
decimator_object = dsp.FIRDecimator(down_sample_factor);
bla = fseek(fid_turbulence2,delay_in_samples*4 ,-1);
while counter<number_of_frames-1
    tic
    
    %get analytic signal and speech current frames:
    current_frame_turbulence1 = fread(fid_turbulence,samples_per_frame,'double');
    current_frame_turbulence2 = fread(fid_turbulence2,samples_per_frame,'double');
    current_frame_speech = fread(fid_speech,samples_per_frame,'double');
    current_frame_turbulence1_delayed = step(fractional_delay_object,current_frame_turbulence1,fractional_delay_needed);
    

    %filter carrier:
%     filtered_carrier = step(carrier_filter_object,current_frame_turbulence1);
%     filtered_carrier2 = step(carrier_filter_object,current_frame_turbulence2);
    
    %scale speech frame:
    current_frame_speech = current_frame_speech/speech_scaling;
    
    %extract analytic signal:
    analytic_signal1 = step(analytic_signal_object,current_frame_turbulence1);
    analytic_signal2 = step(analytic_signal_object,current_frame_turbulence2);
    
    
    current_frame_turbulence1 = (abs(analytic_signal1).^3.*cos(angle(analytic_signal1)) + abs(analytic_signal2).^3.*cos(angle(analytic_signal2)) ) ./ (abs(analytic_signal1).^2 + abs(analytic_signal2).^2);
    analytic_signal1 = step(analytic_signal_object,current_frame_turbulence1);
    analytic_signal2 = step(analytic_signal_object,current_frame_turbulence2);
    
%     figure;
%     subplot(1,2,1);
%     plot(current_frame_turbulence1);
%     subplot(1,2,2);
%     plot(current_frame_turbulence2);
%     close(gcf);
    
%     figure;
%     subplot(1,2,1);
%     plot(abs(analytic_signal1).*cos(angle(analytic_signal1)));
%     subplot(1,2,2); 
%     plot(abs(analytic_signal2).*cos(angle(analytic_signal2)),'g');
%     close(gcf);
    
    %add speech to analytic signal phase:
    turbulence_current_distance = 200;
    turbulence_final_distance = 200;
    turbulence_phase_factor = sqrt(turbulence_final_distance/turbulence_current_distance);
    analytic_signal_combined1 = abs(analytic_signal1).*exp(1i*angle(analytic_signal1)*turbulence_phase_factor).*exp(1i*current_frame_speech);
    analytic_signal_combined2 = abs(analytic_signal2).*exp(1i*angle(analytic_signal2)*turbulence_phase_factor).*exp(1i*current_frame_speech);
    
    %Start Demodulation:
    %(1). create relevant t_vec:
    current_t_vec = (counter-1)*(samples_per_frame/Fs) + (0:samples_per_frame-1)/Fs;
    %(2). multiply by proper phase term to get read of most of carrier:
    analytic_signal_after_carrier_removal1 = analytic_signal_combined1.*exp(-1i*2*pi*Fc*current_t_vec');
    analytic_signal_after_carrier_removal2 = analytic_signal_combined2.*exp(-1i*2*pi*Fc*current_t_vec');
    %(3). Turn analytic signal to FM:
    phase_signal_after_carrier_removal1 = angle(analytic_signal_after_carrier_removal1(2:end).*conj(analytic_signal_after_carrier_removal1(1:end-1)));
    phase_signal_after_carrier_removal2 = angle(analytic_signal_after_carrier_removal2(2:end).*conj(analytic_signal_after_carrier_removal2(1:end-1)));
    %(4). Remove left over DC:
    phase_signal_after_carrier_removal1 = phase_signal_after_carrier_removal1 - mean(phase_signal_after_carrier_removal1);
    phase_signal_after_carrier_removal2 = phase_signal_after_carrier_removal2 - mean(phase_signal_after_carrier_removal2);
    
    if counter>1 
        
        %Add last term from previous frame to create a frame of equal size as original raw frame:
        phase_signal_after_carrier_removal1 = [phase_signal_after_carrier_removal1(1);phase_signal_after_carrier_removal1];
        phase_signal_after_carrier_removal2 = [phase_signal_after_carrier_removal2(1);phase_signal_after_carrier_removal2];
        
        
        click_threshold = 0.01;
        spike_peak_threshold = 2;
        %GET INDIVIDUAL MASKS INDICES TO KEEP:
        indices_to_disregard_click1 = find( (abs(analytic_signal_after_carrier_removal1) < click_threshold));
        indices_to_disregard_phase1 = find( (abs(phase_signal_after_carrier_removal1) > spike_peak_threshold));
        indices_to_disregard_total1 = unique(sort([indices_to_disregard_click1(:);indices_to_disregard_phase1(:)]));
        uniform_sampling_indices1 = (1:length(current_frame))';
        phase_signal_mask1 = (abs(phase_signal_after_carrier_removal1)<spike_peak_threshold);
        analytic_signal_mask1 = (abs(analytic_signal_after_carrier_removal1)>click_threshold);      
        indices_to_disregard_click2 = find( (abs(analytic_signal_after_carrier_removal2) < click_threshold));
        indices_to_disregard_phase2 = find( (abs(phase_signal_after_carrier_removal2) > spike_peak_threshold));
        indices_to_disregard_total2 = unique(sort([indices_to_disregard_click2(:);indices_to_disregard_phase2(:)]));
        uniform_sampling_indices2 = (1:length(current_frame))';
        phase_signal_mask2 = (abs(phase_signal_after_carrier_removal2)<spike_peak_threshold);
        analytic_signal_mask2 = (abs(analytic_signal_after_carrier_removal2)>click_threshold);
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %(5).Remove Clicks:
        flag_interpolate = 1;
        flag_only_interpolate_over_mask = 0; 
        number_of_indices_around_spikes_to_delete_as_well = 0;
        %use despiking if wanted:
        if flag_interpolate==1                                                              
            [phase_signal_corrected11, indices_containing_spikes_expanded1, ~,~,~] = despike_SOL(phase_signal_after_carrier_removal1,flag_only_interpolate_over_mask,0,number_of_indices_around_spikes_to_delete_as_well,3);
            [phase_signal_corrected22, indices_containing_spikes_expanded2, ~,~,~] = despike_SOL(phase_signal_after_carrier_removal2,flag_only_interpolate_over_mask,0,number_of_indices_around_spikes_to_delete_as_well,3);
            indices_to_remove_expanded1 = unique(sort([indices_containing_spikes_expanded1(:);indices_to_disregard_total1(:)]));
            indices_to_remove_expanded2 = unique(sort([indices_containing_spikes_expanded2(:);indices_to_disregard_total2(:)]));
            binary_mask1 = ones(length(phase_signal_after_carrier_removal1),1);
            binary_mask2 = ones(length(phase_signal_after_carrier_removal2),1);
            binary_mask1(indices_to_remove_expanded1) = 0;
            binary_mask2(indices_to_remove_expanded2) = 0;
        end
        
        %Use masks if wanted:
        flag_use_masks = 1;
        if flag_use_masks==1
%             phase_signal_corrected1 = phase_signal_corrected1.*analytic_signal_mask1;
%             phase_signal_corrected1 = phase_signal_corrected1.*phase_signal_mask1;
%             phase_signal_corrected2 = phase_signal_corrected2.*analytic_signal_mask2;
%             phase_signal_corrected2 = phase_signal_corrected2.*phase_signal_mask2;
              phase_signal_after_binary_mask1 = phase_signal_after_carrier_removal1.*binary_mask1;
              phase_signal_after_binary_mask2 = phase_signal_after_carrier_removal2.*binary_mask2;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        
        %(6). Filter signal:
        filtered_phase1 = step(signal_filter_object,phase_signal_after_binary_mask1);
        filtered_phase2 = step(signal_filter_object,phase_signal_after_binary_mask2);
        
        %(7). Turn to PM if wanted
        flag_PM_or_FM = 2;
        if flag_PM_or_FM==1
            filtered_phase1 = cumsum(filtered_phase1); 
            filtered_phase2 = cumsum(filtered_phase2); 
        end
        
        %(8). Combine if wanted:
        flag_combine_before_binary_or_after = 3;
        if flag_combine_before_binary_or_after==1
           rms_per_sample1 = abs(analytic_signal_after_carrier_removal1); 
           rms_per_sample2 = abs(analytic_signal_after_carrier_removal2);
           weight1 = rms_per_sample1.^2 ./ (rms_per_sample1.^2 + rms_per_sample2.^2);
           weight2 = rms_per_sample2.^2 ./ (rms_per_sample1.^2 + rms_per_sample2.^2);
           final_phase = weight1.*filtered_phase1 + weight2.*filtered_phase1;
           final_phase = step(decimator_object,final_phase);
        elseif flag_combine_before_binary_or_after==2
           rms_per_sample1_with_binary = abs(analytic_signal_after_carrier_removal1).*binary_mask1; 
           rms_per_sample2_with_binary = abs(analytic_signal_after_carrier_removal2).*binary_mask2;
           rms_per_sample1_without_binary = abs(analytic_signal_after_carrier_removal1);
           rms_per_sample2_without_binary = abs(analytic_signal_after_carrier_removal2);
           indices_where_both_are_zero = find(rms_per_sample1_with_binary==0 & rms_per_sample2_with_binary==0);
           indices = 1:length(binary_mask1);
           indices_where_at_least_one_isnt_zero = setdiff(indices,indices_where_both_are_zero);
           weight1 = zeros(length(binary_mask1),1);
           weight2 = zeros(length(binary_mask1),1);
           weight1(indices_where_both_are_zero) = rms_per_sample1_without_binary(indices_where_both_are_zero).^2 ./ (rms_per_sample1_without_binary(indices_where_both_are_zero).^2+rms_per_sample2_without_binary(indices_where_both_are_zero).^2);
           weight2(indices_where_both_are_zero) = rms_per_sample2_without_binary(indices_where_both_are_zero).^2 ./ (rms_per_sample1_without_binary(indices_where_both_are_zero).^2+rms_per_sample2_without_binary(indices_where_both_are_zero).^2);
           weight1(indices_where_at_least_one_isnt_zero) = rms_per_sample1_with_binary(indices_where_at_least_one_isnt_zero).^2 ./ (rms_per_sample1_with_binary(indices_where_at_least_one_isnt_zero).^2+rms_per_sample2_with_binary(indices_where_at_least_one_isnt_zero).^2);
           weight2(indices_where_at_least_one_isnt_zero) = rms_per_sample2_with_binary(indices_where_at_least_one_isnt_zero).^2 ./ (rms_per_sample1_with_binary(indices_where_at_least_one_isnt_zero).^2+rms_per_sample2_with_binary(indices_where_at_least_one_isnt_zero).^2);
           final_phase = weight1.*filtered_phase1 + weight2.*filtered_phase1;
           final_phase = step(decimator_object,final_phase);
        else
           filtered_phase1 = step(signal_filter_object,phase_signal_after_carrier_removal1);
           filtered_phase2 = step(signal_filter_object,phase_signal_after_carrier_removal2);
        end
        
        %(9). Down sample:
        filtered_phase1 = step(decimator_object,filtered_phase1);
        filtered_phase2 = step(decimator_object,filtered_phase2);
%         
%         if flag_save_to_wav==1
%             step(audio_file_writer_demodulated,[filtered_phase(:),filtered_phase(:)]*multiplication_factor);
%         end
        
        flag_sound_demodulation=1;
        if flag_sound_demodulation==1
%             step(audio_player_object,[filtered_phase1(:),filtered_phase1(:)]*multiplication_factor);
%             step(audio_player_object,[filtered_phase1(:)]*multiplication_factor);
            step(audio_player_object,[filtered_phase2(:)]*multiplication_factor);
%             step(audio_player_object,[final_phase(:)]*multiplication_factor);
        end
        
    end
    
    counter=counter+1;
    toc
end



try
    fclose(fid_turbulence);
    fclose(fid_turbulence2);
    fclose(fid_speech); 
    release(audio_player_object);
    release(audio_file_writer_demodulated);
    release(analytic_signal_object);
    fclose('all');
catch 
end








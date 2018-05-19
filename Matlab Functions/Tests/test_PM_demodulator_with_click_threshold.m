%test_PM_demodulator_with_click_threshold:

% function [] = simple_PM_demodulator(directory,file_name,flag_bin_or_wav,flag_save_to_wav,flag_sound_demodulation,multiplication_factor)
%simple PM demodulator based on dsp objects:
clear all;
file_name='shirt_2mm_ver_200m_audio';
directory='C:\Users\master\Desktop\matlab';
flag_save_to_wav=1;
flag_sound_demodulation=1;

file_name_with_bin = strcat(file_name,'.bin');
full_file_name = fullfile(directory,file_name_with_bin);
 
try
    fclose(fid); 
    release(audio_player_object); 
    release(audio_file_writer_demodulated);
    release(audio_file_reader);
    release(analytic_signal_object);  
    fclose('all');
catch 
end

%audio reader object:
fid = fopen(full_file_name,'r');
number_of_elements_in_file = fread(fid,1,'double');
Fs = fread(fid,1,'double');
final_Fs_to_sound = 44000;
down_sample_factor = round(Fs/final_Fs_to_sound);
Fs_downsampled = Fs/down_sample_factor;


%Decide on number of samples per frame:
time_for_FFT_frame = 1; %10[mSec]
counts_per_time_frame = round(Fs*time_for_FFT_frame);
samples_per_frame = counts_per_time_frame;
number_of_frames_in_file = floor(number_of_elements_in_file/samples_per_frame);
number_of_seconds_in_file = floor(number_of_elements_in_file/Fs);

%Default Initial Values:
flag_PM_or_FM=1;
if flag_PM_or_FM==1
   PM_FM_str='PM'; 
else
   PM_FM_str='FM';
end
    
%audio player object:
audio_player_object = dsp.AudioPlayer;
audio_player_object.SampleRate=44100;
audio_player_object.QueueDuration = 2;
 
%analytic signal object:
analytic_signal_object = dsp.AnalyticSignal;
analytic_signal_object.FilterOrder=100;

%basic filter parameters:
carrier_filter_parameter = 20;
signal_filter_parameter = 20;
carrier_filter_length = 128*1; 
signal_filter_length = 128*10; 
%signal filter: 
filter_name_signal = 'hann';
signal_filter_type = 'bandpass';
signal_start_frequency = 150;
signal_stop_frequency = 3000;
[signal_filter] = get_filter_1D(filter_name_signal,signal_filter_parameter,signal_filter_length,Fs_downsampled,signal_start_frequency,signal_stop_frequency,signal_filter_type);
signal_filter_object = dsp.FIRFilter('Numerator',signal_filter.Numerator);    
signal_filter_object2 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
signal_filter_object3 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
signal_filter_object4 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
signal_filter_object5 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
signal_filter_object6 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
%carrier filter:
Fc = 12000; %initial Fc
BW = signal_stop_frequency*2;
filter_name_carrier = 'hann';
f_low_cutoff_carrier = Fc-BW/2; 
f_high_cutoff_carrier = Fc+BW/2;
carrier_filter_type = 'bandpass';
[carrier_filter] = get_filter_1D(filter_name_carrier,carrier_filter_parameter,carrier_filter_length,Fs,f_low_cutoff_carrier,f_high_cutoff_carrier,carrier_filter_type);
carrier_filter_object = dsp.FIRFilter('Numerator',carrier_filter.Numerator);

%Audio Equalizer:
%(1). simply a frequency gain function interpolated:
full_frequency_vec = fft_get_frequency_vec(samples_per_frame,Fs,0);
number_of_frequency_bands = 8;
final_frequency = 3200;
frequency_bands = linspace(0,final_frequency,number_of_frequency_bands);
frequency_bands_spacing = frequency_bands(2)-frequency_bands(1);
frequency_bands_gains_dB = zeros(8,1);
frequency_bands_gains_dB(1) = 0;
frequency_bands_gains_dB(2) = -10;
frequency_bands_gains_dB(3) = 10;
frequency_bands_gains_dB(4) = -5;
frequency_bands_gains_dB(5) = 5;
frequency_bands_gains_dB(6) = -1; 
frequency_bands_gains_dB(7) = 0;
frequency_bands_gains_dB(8) = 0; 
%(2). a series of band pass IIR filters:
%filter parameters:
bandpss_ripple_dB = 0.5;
stopband_attenuation_dB = 30;
filter_order = 15;
equalizer_filters = zeros(number_of_frequency_bands,filter_order);
% for band_counter=1:number_of_frequency_bands-1
%     %create filter:
%     if band_counter==1
%        %lowpass:
%        equalizer_filters(band_counter,:) = cheby1(filter_order,bandpass_ripple,frequency_bands_gains_dB(band_counter+1),'low');
%     elseif band_counter<number_of_frequency_bands
%        %bandpass:
%        equalizer_filters(band_counter,:) = cheby1(filter_order,bandpass_ripple,[frequency_bands_gains_dB(band_counter-1),frequency_bands_gains_dB(band_counter+1)],'pass');
%     elseif band_counter==number_of_frequency_bands-1
%        %highpass:
%        equalizer_filters(band_counter,:) = cheby1(filter_order,bandpass_ripple,frequency_bands_gains_dB(band_counter-1),'high');
%     end
% end
% %get final non-filter transfer function:
% frequency_bands = linspace(-final_frequency,final_frequency,2*number_of_frequency_bands);
% frequency_bands_gains_dB_two_sided = [flip(frequency_bands_gains_dB),frequency_bands_gains_dB];
% frequency_bands_gains_linear = 10.^(frequency_bands_gains_dB_two_sided/10);
% frequency_bands_gains_linear_interpolated = interp1(frequency_bands,frequency_bands_gains_linear,full_frequency_vec,'cubic');
% full_frequency_vec = fft_get_frequency_vec(samples_per_frame,Fs,0);



%save demodulation to wav file if wanted:
if flag_save_to_wav==1
    audio_file_writer_demodulated = dsp.AudioFileWriter;
    audio_file_writer_demodulated.Filename = strcat(fullfile(directory,file_name),'  ' , PM_FM_str ,' final demodulated audio ', ' ', num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),...
        '[Hz] diff phase','.wav');
    audio_file_writer_demodulated.SampleRate = 44100;
end

%Get detector frequency calibration:
load('detector_frequency_calibration');
x_frequency = f_vec;
y_frequency = fft_sum_smoothed;
x_calibration = fft_get_frequency_vec(samples_per_frame,Fs,0);
y_calibration = interp1(x_frequency,y_frequency,x_calibration); 
  
counter=1;
multiplication_factor=1; 
click_threshold = 0.008; %default 0.008
spike_peak_threshold = 12; %default 0.1
%Jump to start_second:
start_second = 10; 
stop_second = start_second+20;
stat_second = max(start_second,0);
stop_second = min(stop_second,number_of_seconds_in_file);
number_of_samples_to_read = floor((stop_second-start_second)*Fs/samples_per_frame);
fseek(fid,8*ceil(start_second*Fs),-1);
variance_over_time = 0;
filtered_phase_over_time = 0;
last_phase_previous = 0;
final_final = 0;
last_cumsum = 0;
phase_signal_difference = zeros(samples_per_frame,1);
while counter<number_of_samples_to_read
    tic
 
    %Read current frame:
    current_frame = fread(fid,samples_per_frame,'double');
     
    %Adjust frequencies according to detector calibration:
%     tic
    current_fft = fftshift(fft(current_frame));
    current_fft = current_fft./y_calibration';
    current_frame = real(ifft(fftshift(current_fft)));
%     toc
    
    %filter carrier:
%     tic
    flag_filter_carrier = 1;
    if flag_filter_carrier == 1
        filtered_carrier = step(carrier_filter_object,current_frame);
    else
        filtered_carrier = current_frame;
    end
%     toc
    
    %extract analytic signal:
    analytic_signal = step(analytic_signal_object,filtered_carrier);
    
    %USE DAN'S DEMODULATINO METHOD:
    %(1). creating relevant t_vec:
    current_t_vec = (counter-1)*(samples_per_frame/Fs) + (0:samples_per_frame-1)/Fs;
    %(2). multiply by proper phase term to get read of most of carrier:
    analytic_signal_after_carrier_removal = analytic_signal.*exp(-1i*2*pi*Fc*current_t_vec');
    %(3). Turn analytic signal to FM:
    phase_signal_difference(2:end) = angle(analytic_signal_after_carrier_removal(2:end).*conj(analytic_signal_after_carrier_removal(1:end-1)));
    
    %Continuity from last frame (its possible to just use gradient but it's
    %a bit slower and i don't like gradient:
    first_phase_current = angle(analytic_signal_after_carrier_removal(1));
    phase_signal_difference(1) = first_phase_current-last_phase_previous;
    last_phase_previous = angle(analytic_signal_after_carrier_removal(end));
    
    %Set thresholds and Build masks:
    click_threshold = 0.002; %silence where analytic signal amplitude is below this threshold
    spike_peak_threshold = 20; %silence where phase is above this threshold
    %Get individual mask indices to get rid of:
    indices_to_disregard_click = find( abs(analytic_signal_after_carrier_removal) < click_threshold );
    indices_to_disregard_phase = find( abs(phase_signal_difference) > spike_peak_threshold );
    indices_to_disregard_total = unique(sort([indices_to_disregard_click(:);indices_to_disregard_phase(:)]));
    uniform_sampling_indices = (1:length(current_frame))';
    indices_to_keep_total = ismember(uniform_sampling_indices, indices_to_disregard_total);
    
    %FIND ANALYTIC SIGNAL AND PHASE SIGNAL MASKS:
    phase_signal_mask = (abs(phase_signal_difference)<spike_peak_threshold);
    analytic_signal_mask = (abs(analytic_signal_after_carrier_removal)>click_threshold);
        
    %Remove Clicks options:
    flag_use_only_found_only_additional_or_both = 3;
    additional_indices_to_interpolate_over = indices_to_disregard_total;
    number_of_indices_around_spikes_to_delete_as_well = 0;
    flag_use_my_spline_or_matlabs_or_binary_or_do_nothing = 3;
      
    %(5). Remove clicks: 
    despike_sensitivity_factor = 1;
    flag_use_despiking_or_just_my_mask_or_nothing = 1;
    if flag_use_despiking_or_just_my_mask_or_nothing == 1
%         [phase_signal_after_mask, indices_containing_spikes_expanded, ~,~,~] = despike_SOL(phase_signal_difference', flag_use_only_found_only_additional_or_both, additional_indices_to_interpolate_over, number_of_indices_around_spikes_to_delete_as_well, flag_use_my_spline_or_matlabs_or_binary_or_do_nothing); 
        [phase_signal_after_mask, indices_containing_spikes_expanded, ~,~] = despike_SOL_fast_with_logical_masks(phase_signal_difference, flag_use_only_found_only_additional_or_both, additional_indices_to_interpolate_over, number_of_indices_around_spikes_to_delete_as_well, flag_use_my_spline_or_matlabs_or_binary_or_do_nothing,despike_sensitivity_factor); 
%         [phase_signal_after_mask] = despikeADV(phase_signal_difference);
%         phase_signal_after_mask = phase_signal_after_mask.*analytic_signal_mask;
    elseif flag_use_despiking_or_just_my_mask_or_nothing == 2
        phase_signal_after_mask = phase_signal_difference.*analytic_signal_mask;
        phase_signal_after_mask = phase_signal_after_mask.*phase_signal_mask;
    elseif flag_use_despiking_or_just_my_mask_or_nothing == 3
        phase_signal_after_mask = phase_signal_difference;
    end
     
    flag_show_after_mask=0;
    if flag_show_after_mask==1
       figure
       plot(phase_signal_difference);
       hold on;
       plot(phase_signal_after_mask,'g');
       close(gcf);
    end 
      
    %(6). down sample:
    phase_signal_after_mask = downsample(phase_signal_after_mask(:),down_sample_factor);

    %(7). filter demodulated signal:
    filtered_phase = step(signal_filter_object,phase_signal_after_mask(:));
%     filtered_phase = irlssmooth(filtered_phase,20,0);
%     filtered_phase = powersmooth(filtered_phase,0,1);
%     filtered_phase = smooth(filtered_phase,15,'lowess');
%     filtered_phase = smoothn(filtered_phase,25);
    
    
%     figure
%     bla = smoothn(filtered_phase,15); 
%     plot(filtered_phase);
%     hold on;
%     plot(bla,'g');
%     close(gcf);
    
    %(8). Turn to PM if wanted
    flag_PM_or_FM = 1; 
    if flag_PM_or_FM==1;
        filtered_phase = step(signal_filter_object2,filtered_phase);
        filtered_phase = cumsum(filtered_phase)+last_cumsum;
        last_cumsum = filtered_phase(end);
    end 
    
    %Assign final signal:
    final_signal = filtered_phase;
%     final_signal = real(ifft(ifftshift(fftshift(fft(final_signal)).*frequency_bands_gains_linear_interpolated')));
%     tic
%     final_signal = real(ifft(fft(final_signal)));
%     toc
    
    %Sound carrier AM:
    flag_use_carrier_AM_or_DC_or_none=3;
    if flag_use_carrier_AM_or_DC_or_none==1
        final_signal2 = abs(analytic_signal_after_carrier_removal(:));
        final_signal2 = step(signal_filter_object3,final_signal2(:));
        final_signal2 = cumsum(final_signal2(:));
        final_signal2 = step(signal_filter_object4,final_signal2(:));
        final_signal = final_signal + 4*final_signal2;
        
    elseif flag_use_carrier_AM_or_DC_or_none==2
        final_signal2 = step(signal_filter_object3,current_frame(:));
        final_signal2 = (final_signal2(:));
        final_signal2 = step(signal_filter_object4,final_signal2(:));
        final_signal = final_signal2;
        
    else
        final_signal = final_signal;
    end
    
    final_final = [final_final(:);final_signal];
    
    flag_sound_demodulation = 1;
    final_signal = final_signal*1/5;
    if flag_sound_demodulation==1 && counter>1
        step(audio_player_object,[final_signal(:),final_signal(:)]);
    end
    
    %Save to .wav file if wanted:
    flag_save_to_wav = 0;
    if flag_save_to_wav==1
        step(audio_file_writer_demodulated,final_signal(:));
    end
    
    counter=counter+1;
    toc
end



try
    fclose(fid);
    fclose('all');
    release(audio_player_object);
    release(audio_file_writer_demodulated);
    release(audio_file_reader);
    release(analytic_signal_object);
catch 
end















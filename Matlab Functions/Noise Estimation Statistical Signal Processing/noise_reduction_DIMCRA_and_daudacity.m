function [final_signal] = noise_reduction_DIMCRA_and_daudacity(file_name)
%audacity + DIMCRA noise reduction:
global noise_sensitivity_factor noise_attenuation_factor
global attack_decay_per_block release_decay_per_block
global gains_history_number_of_blocks
global center_of_search_window_in_gains_history start_of_search_window_in_gains_history stop_of_search_window_in_gains_history
global spectrums_history ffts_history gains_history  frequency_index_to_cutoff_algorithm_memory
global gains_history_full_flipped_indices current_search_window_indices_within_gains_history 
global current_fft_update_index current_final_fft_index center_of_search_window_index
global FFT_size_signal_filter  
global noise_gate_threshold_vec speech_probability_history classical_gain_history

%Get wav file to denoise:
% file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio150-3000[Hz]';
% file_name = 'Sound000.wav';
file_name = '1500m, reference loudness=0dBspl, speakers, laser power=7.8 , Fs=4935Hz, channel1.wav';
% file_name = 'C:\Users\master\Desktop\matlab\parameter experiment sound files\counting forward';
[input_signal,Fs] = wavread(file_name);
input_signal = input_signal(:,1); 
% input_signal = input_signal(1:4935*20);
input_signal = resample(input_signal,8000,Fs);
% input_signal = add_noise_of_certain_SNR(input_signal,0,1,-0.25);
Fs = 8000; 
 
%Initialize audio framing parameters: 
samples_per_frame = 2048; 
overlap_samples_per_frame = (samples_per_frame * 2/4);	
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;

%Initialize buffer object:
buffer_object = dsp.Buffer(samples_per_frame , overlap_samples_per_frame);

%Get frame window:
[frame_window] = get_approximately_COLA_window(samples_per_frame,overlap_samples_per_frame);

%Initialize audacity style noise reduction default parameters:
number_of_audacity_channels = 1;
noise_estimation_method = 'dimcra';
noise_attenuation_in_dB = -50;
sensitivity_in_dB = 6; 
frequency_smoothing_BW_in_Hz = 0;
attack_time = 0.05; 
release_time = 0.05; 
spectral_activity_search_time = 0.1;
noise_sensitivity_factor = 10^(sensitivity_in_dB/10);
noise_attenuation_factor = 10^(noise_attenuation_in_dB/20);


%Signal BandPass filter parameters:
signal_filter_parameter = 7;
signal_filter_length = 128*8-2; 
signal_filter_window_type = 'kaiser';
signal_filter_type = 'bandpass';
signal_filter_start_frequency = 20;
signal_filter_stop_frequency = 2400;
%create filter:
[signal_filter] = get_filter_1D(signal_filter_window_type,signal_filter_parameter,signal_filter_length,Fs,signal_filter_start_frequency,signal_filter_stop_frequency,signal_filter_type);
%create fft_overlap_add_object:
[signal_filter_fft_object] = fft_overlap_add_object(signal_filter,Fs,samples_per_frame,overlap_samples_per_frame,2,0);
%signal filter FFT parameters:
FFT_size_signal_filter = signal_filter_fft_object.FFT_size;
frequency_index_to_cutoff_algorithm_memory = round(signal_filter_stop_frequency/(Fs/2)*(FFT_size_signal_filter/2));
signal_filter_fft = signal_filter_fft_object.filter_fft;

%Initialize needed variables:
last_phase_previous = 0;
noise_gate_threshold_vec = zeros(frequency_index_to_cutoff_algorithm_memory,1);
current_sub_frame = zeros(samples_per_frame,1);
last_cumsum = 0;
 
%Noise reduction Parameters:
number_of_frequency_smoothing_bins = int8(round((frequency_smoothing_BW_in_Hz/(Fs/2)) * non_overlapping_samples_per_frame));
attack_decay_number_of_samples = attack_time*Fs;
attack_decay_number_of_blocks = int8(1 + round( attack_decay_number_of_samples / non_overlapping_samples_per_frame ));
attack_decay_per_block = 10^( noise_attenuation_in_dB/(20*double(attack_decay_number_of_blocks)) );
release_decay_number_of_samples = release_time*Fs;
release_decay_number_of_blocks = int8(1 + round( release_decay_number_of_samples / non_overlapping_samples_per_frame ));
release_decay_per_block = 10^( noise_attenuation_in_dB/(20*double(release_decay_number_of_blocks)) );
spectral_minimum_search_number_of_blocks = int8(round( spectral_activity_search_time * Fs / non_overlapping_samples_per_frame ));
spectral_minimum_search_number_of_blocks = int8(max( 2 , spectral_minimum_search_number_of_blocks ));
gains_history_number_of_blocks = attack_decay_number_of_blocks + release_decay_number_of_blocks + 1;
gains_history_number_of_blocks = int8(max(gains_history_number_of_blocks,spectral_minimum_search_number_of_blocks));

%Search window history indices:
center_of_search_window_in_gains_history = int8(1 + release_decay_number_of_blocks);
start_of_search_window_in_gains_history = int8(center_of_search_window_in_gains_history - floor(spectral_minimum_search_number_of_blocks/2));
stop_of_search_window_in_gains_history = int8(center_of_search_window_in_gains_history + ceil(spectral_minimum_search_number_of_blocks/2));

%Initialize audacity algorithm running indices and indices vecs:
gains_history_full_flipped_indices = int8(flip(1:1:gains_history_number_of_blocks)');
current_search_window_indices_within_gains_history = int8(gains_history_full_flipped_indices(start_of_search_window_in_gains_history:stop_of_search_window_in_gains_history));
current_fft_update_index = int8(gains_history_full_flipped_indices(end));
current_final_fft_index = int8(gains_history_full_flipped_indices(end-1));
center_of_search_window_index = int8(gains_history_full_flipped_indices(center_of_search_window_in_gains_history));

%Multiply variables to many channels:
gains_history_full_flipped_indices = repmat(gains_history_full_flipped_indices,[1,number_of_audacity_channels]);
current_search_window_indices_within_gains_history = repmat(current_search_window_indices_within_gains_history,[1,number_of_audacity_channels]);
current_fft_update_index = repmat(current_fft_update_index,[1,number_of_audacity_channels]);
current_final_fft_index = repmat(current_final_fft_index,[1,number_of_audacity_channels]);
center_of_search_window_index = repmat(center_of_search_window_index,[1,number_of_audacity_channels]);

%Initialize Matrices containing spectrums and gains history:
spectrums_history = zeros(frequency_index_to_cutoff_algorithm_memory,gains_history_number_of_blocks,number_of_audacity_channels);
ffts_history = zeros(FFT_size_signal_filter,gains_history_number_of_blocks,number_of_audacity_channels);
gains_history = ones(frequency_index_to_cutoff_algorithm_memory,gains_history_number_of_blocks,number_of_audacity_channels) * noise_attenuation_factor;
speech_probability_history = zeros(frequency_index_to_cutoff_algorithm_memory,gains_history_number_of_blocks,number_of_audacity_channels);
classical_gain_history  = zeros(frequency_index_to_cutoff_algorithm_memory,gains_history_number_of_blocks,number_of_audacity_channels); 

%Initialize noise reduction flag:
flag_initialize_noise_estimation = 1;
 
%Initialize audio player object:
audio_player_object = dsp.AudioPlayer;
audio_player_object.SampleRate = Fs;

%Loop Over Frames and Denoise Them:
start_index = 1;
sub_frame_counter = 1;
number_of_frames = floor(length(input_signal)/non_overlapping_samples_per_frame);
noise_estimation_parameters = 0;
while sub_frame_counter <= number_of_frames
        
        %Get Current frame, fft, and spectrum:
        stop_index = start_index + non_overlapping_samples_per_frame - 1;
        current_new_samples = input_signal(start_index:stop_index);
        current_sub_frame = step(buffer_object,current_new_samples);
        current_sub_frame_fft2 = fft(filter([1,-0.95],1,current_sub_frame).*frame_window,FFT_size_signal_filter);
        current_sub_frame = current_sub_frame .* frame_window;
        current_sub_frame_fft = fft(current_sub_frame,FFT_size_signal_filter);
        
        
        current_sub_frame_spectrum = abs(current_sub_frame_fft2(1:frequency_index_to_cutoff_algorithm_memory)).^2;
        
        %Update Noise Profile Automatically:
        [noise_estimation_parameters,noise_gate_threshold_vec,flag_initialize_noise_estimation] = ...
            audio_estimate_noise_profile_parameters_united(current_sub_frame_spectrum,noise_estimation_method,noise_estimation_parameters,Fs,flag_initialize_noise_estimation);
        
        %Use audacity filter:
        flag_use_speech_probability_or_not = 1;
        flag_use_one_mean_two_mean_or_speech_probability_as_gain = 1;
        [fft_after_gain_function,current_gain_function] = audacity_filter(1,current_sub_frame_fft,current_sub_frame_spectrum,noise_estimation_parameters.speech_presence_probability,noise_estimation_parameters.GH1, flag_use_speech_probability_or_not , flag_use_one_mean_two_mean_or_speech_probability_as_gain);
          
        %use signal filter:
        [valid_filtered_signal] = signal_filter_fft_object.filter(fft_after_gain_function,2);
        
        %Integrate final signal and filter it (PM) or sound direct signal (FM):
        flag_PM_or_FM = 2;
        if flag_PM_or_FM==1
            final_signal = cumsum(valid_filtered_signal)+last_cumsum;
            last_cumsum = final_signal(end);
        else
            final_signal = valid_filtered_signal;
        end 
        
        %Sound Demodulation
        speech_factor = 5;
        flag_sound_demodulation = 1;
        if flag_sound_demodulation==1
            step(audio_player_object,final_signal(:)*speech_factor);
        end 
        
%         subplot(2,1,1)
%         plot(noise_estimation_parameters.speech_presence_probability);
%         subplot(2,1,2)
%         plot(noise_estimation_parameters.noise_ps,'b');
%         hold on;
%         plot(current_sub_frame_spectrum,'g');
%         hold off;
%         drawnow
        
        %Increment indices in gains history indices:
        gains_history_full_flipped_indices = mod(gains_history_full_flipped_indices,gains_history_number_of_blocks)+1;
       
        %Increment sub frame counter:
        sub_frame_counter = sub_frame_counter+1;
        start_index = start_index + non_overlapping_samples_per_frame;
end  %sub frames loop





function [fft_after_gain_function,current_gain_function] = audacity_filter(channel_counter,current_sub_frame_fft,current_sub_frame_spectrum,speech_presence_probability,classical_gain_function,flag_use_speech_probability_or_not,flag_use_one_mean_two_mean_or_speech_probability_as_gain)
global noise_sensitivity_factor noise_attenuation_factor
global attack_decay_per_block release_decay_per_block
global gains_history_number_of_blocks
global center_of_search_window_in_gains_history start_of_search_window_in_gains_history stop_of_search_window_in_gains_history
global spectrums_history ffts_history gains_history  frequency_index_to_cutoff_algorithm_memory
global gains_history_full_flipped_indices current_search_window_indices_within_gains_history 
global current_fft_update_index current_final_fft_index center_of_search_window_index
global FFT_size_signal_filter  
global noise_gate_threshold_vec speech_probability_history classical_gain_history
% global current_sub_frame_fft current_sub_frame_spectrum

%Update histories:
ffts_history(:,current_fft_update_index(channel_counter),channel_counter) = current_sub_frame_fft;
spectrums_history(:,current_fft_update_index(channel_counter),channel_counter) = current_sub_frame_spectrum;
gains_history(:,current_fft_update_index(channel_counter),channel_counter) = noise_attenuation_factor;
speech_probability_history(:,current_fft_update_index(channel_counter,channel_counter)) = speech_presence_probability;
classical_gain_history(:,current_fft_update_index(channel_counter,channel_counter)) = classical_gain_function;

%Find Indices above noise and get Gain in those indices:
if flag_use_speech_probability_or_not==0
    
    noise_classification_flag_counter=1;
    if noise_classification_flag_counter==1
        %check if second greatest in search window is above noise threshold
        current_sorted_search_window_spectrums = sort(spectrums_history(:,current_search_window_indices_within_gains_history,channel_counter),2);
        indices_above_noise_gate = uint16(find(current_sorted_search_window_spectrums(:,end-1) > noise_sensitivity_factor * noise_gate_threshold_vec));
    elseif noise_classification_flag_counter==2
        %check if current window mean is above noise threshold
        current_mean_search_window_spectrums = mean(spectrums_history(:,current_search_window_indices_within_gains_history,channel_counter),2);
        indices_above_noise_gate = uint16(find(current_mean_search_window_spectrums > noise_sensitivity_factor * noise_gate_threshold_vec));
    end
elseif flag_use_speech_probability_or_not==1
    
    speech_probability_vec = speech_probability_history(:,current_search_window_indices_within_gains_history,channel_counter);
    classical_gain_vec = classical_gain_history(:,current_search_window_indices_within_gains_history,channel_counter);
    
    [speech_probability_history] = smooth_mat_by_row_or_column_value_decay(...
    speech_probability_history,...
    attack_decay_per_block,...
    release_decay_per_block,...
    gains_history_full_flipped_indices,...
    center_of_search_window_in_gains_history,...
    center_of_search_window_index,...
    noise_attenuation_factor,...
    channel_counter);
    
	averaged_speech_probability = speech_probability_history(:,current_final_fft_index(channel_counter),channel_counter);

    averaged_speech_probability = mean(speech_probability_vec,2);
    
    if flag_use_one_mean_two_mean_or_speech_probability_as_gain==1
        averaged_GH = mean(classical_gain_vec,2);
        final_gain_vec = averaged_GH.^(averaged_speech_probability) .* (noise_attenuation_factor*ones(size(averaged_speech_probability))).^(1-averaged_speech_probability);
    elseif flag_use_one_mean_two_mean_or_speech_probability_as_gain==2
        final_gain_vec = mean(mean(classical_gain_vec.^(speech_probability_vec) .*(noise_attenuation_factor*ones(size(speech_probability_vec))).^(1-speech_probability_vec),2)  ,  2);
    else
        final_gain_vec = averaged_speech_probability;  
    end

    %only update indices where averaged speech probability is greater than zero:
    indices_above_noise_gate = uint16(find(averaged_speech_probability>0));
    final_gain_vec = final_gain_vec(indices_above_noise_gate);
end
   
%DECAY THE GAIN IN BOTH DIRECTIONS:
gains_history_above_noise = gains_history(indices_above_noise_gate,:,channel_counter);   
if flag_use_speech_probability_or_not==1
   values_to_raise_to = final_gain_vec; 
else
   values_to_raise_to = 1;
end
[gains_history] = raise_and_decay_forward_and_backward_in_time(...
    gains_history,...
    indices_above_noise_gate,...
    values_to_raise_to,...
    attack_decay_per_block,...
    release_decay_per_block,...
    gains_history_full_flipped_indices,...
    center_of_search_window_in_gains_history,...
    center_of_search_window_index,...
    noise_attenuation_factor,...
    channel_counter);

%Get current gain function
current_gain_function = noise_attenuation_factor*ones(FFT_size_signal_filter,1);
current_gain_function(1:frequency_index_to_cutoff_algorithm_memory) = gains_history(:,current_final_fft_index(channel_counter),channel_counter);
current_gain_function(end/2+1:end/2+1+frequency_index_to_cutoff_algorithm_memory-1) = gains_history(:,current_final_fft_index(channel_counter),channel_counter);

%Multiply current fft by gain function:
fft_after_gain_function = ffts_history(:,current_final_fft_index(channel_counter),channel_counter).*current_gain_function + eps*1i;

%Increment indices in gains history indices:
gains_history_full_flipped_indices(:,channel_counter) = mod(gains_history_full_flipped_indices(:,channel_counter),gains_history_number_of_blocks)+1;
current_fft_update_index(channel_counter) = gains_history_full_flipped_indices(end,channel_counter);
current_final_fft_index(channel_counter) = gains_history_full_flipped_indices(end-1,channel_counter);
center_of_search_window_index(channel_counter) = gains_history_full_flipped_indices(center_of_search_window_in_gains_history,channel_counter);
current_search_window_indices_within_gains_history = gains_history_full_flipped_indices(start_of_search_window_in_gains_history:stop_of_search_window_in_gains_history,channel_counter);

        










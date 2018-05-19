%test_PM_demodulator_with_click_threshold:

% function [] = simple_PM_demodulator(directory,file_name,flag_bin_or_wav,flag_save_to_wav,flag_sound_demodulation,multiplication_factor)
%simple PM demodulator based on dsp objects:
% clear all;
file_name='shirt_2mm_ver_200m_audio';
directory='C:\Users\master\Desktop\matlab';
flag_save_to_wav=1;
flag_sound_demodulation=1;

file_name_with_bin = strcat(file_name,'.bin');
full_file_name = fullfile(directory,file_name_with_bin);

%  profile clear;
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
% time_for_FFT_frame = 0.045; %10[mSec]
% counts_per_time_frame = round(Fs*time_for_FFT_frame);
% samples_per_frame = counts_per_time_frame;
samples_per_frame = 2048;
global_to_sub_frame_size_ratio = 40;
samples_per_frame_demodulation = samples_per_frame * global_to_sub_frame_size_ratio;
number_of_frames_in_file = floor(number_of_elements_in_file/samples_per_frame);
number_of_seconds_in_file = floor(number_of_elements_in_file/Fs);
full_indices_vec = (1:samples_per_frame_demodulation)';

%Default Initial Values:
flag_PM_or_FM=1;
if flag_PM_or_FM==1
   PM_FM_str='PM'; 
else
   PM_FM_str='FM';
end
 
%analytic signal object:
analytic_signal_object = dsp.AnalyticSignal;
analytic_signal_object.FilterOrder=100;

%basic filter parameters:
carrier_filter_parameter = 10;
signal_filter_parameter = 7;
carrier_filter_length = 128*1; 
signal_filter_length = 128*8-2; 
%signal filter: 
filter_name_signal = 'kaiser';
signal_filter_type = 'bandpass';
signal_start_frequency = 170;
signal_stop_frequency = 3500;
[signal_filter] = get_filter_1D(filter_name_signal,signal_filter_parameter,signal_filter_length,Fs_downsampled,signal_start_frequency,signal_stop_frequency,signal_filter_type);
signal_filter_object = dsp.FIRFilter('Numerator',signal_filter.Numerator);    
signal_filter_object2 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
signal_filter_object3 = dsp.FIRFilter('Numerator',signal_filter.Numerator);

%carrier filter:
Fc = 12000; %initial Fc
BW = signal_stop_frequency*2;
filter_name_carrier = 'hann';
f_low_cutoff_carrier = Fc-BW/2;
f_high_cutoff_carrier = Fc+BW/2;
carrier_filter_type = 'bandpass';
[carrier_filter] = get_filter_1D(filter_name_carrier,carrier_filter_parameter,carrier_filter_length,Fs,f_low_cutoff_carrier,f_high_cutoff_carrier,carrier_filter_type);
carrier_filter_object = dsp.FIRFilter('Numerator',carrier_filter.Numerator);

%get FFT_size based on carrier filter and signal filter lengths:
FFT_size_carrier_filter = 2^nextpow2(samples_per_frame_demodulation+carrier_filter_length-1);
FFT_size_signal_filter = 2^nextpow2(samples_per_frame+signal_filter_length-1);

%save demodulation to wav file if wanted:
if flag_save_to_wav==1
    audio_file_writer_demodulated = dsp.AudioFileWriter;
    audio_file_writer_demodulated.Filename = strcat(fullfile(directory,file_name),'  ' , PM_FM_str ,' final demodulated audio ', ' ', num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),'[Hz]'...
        ,'lsq smoother without deframing2.wav');
    audio_file_writer_demodulated.SampleRate = 44100;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NOISE REMOVAL STUFF:

%Frame parameters:
overlap_samples_per_frame = (samples_per_frame * 1/2);	
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
hanning_window = make_column(hanning(samples_per_frame));
enhanced_speech_overlapped_samples_from_previous_frame = zeros(overlap_samples_per_frame,1);

%Dewindowing:
hamming_window = make_column(hamming(samples_per_frame));
hanning_window_for_frame_edge_averaging = make_column(hanning(2*overlap_samples_per_frame-1));
deframing_window = [hanning_window_for_frame_edge_averaging(1:overlap_samples_per_frame);ones(samples_per_frame-2*overlap_samples_per_frame,1);...
    hanning_window_for_frame_edge_averaging(overlap_samples_per_frame:end)]./hamming_window;

%Input Constant Parameters:
noise_attenuation_in_dB = -10;
sensitivity_in_dB = 6; 
frequency_smoothing_BW_in_Hz = 0;
number_of_frequency_bins = round(frequency_smoothing_BW_in_Hz/(1/(samples_per_frame/Fs)));
% number_of_frequency_bins = 4;
attack_time = 0.02; 
release_time = 0.1; 
spectral_minimum_search_time = 0.05;
sensitivity_factor = 10^(sensitivity_in_dB/10);
noise_attenuation_factor = 10^(noise_attenuation_in_dB/20);
spectral_floor_factor = 0.05;

%Noise reduction Parameters:
number_of_frequency_smoothing_bins = round((frequency_smoothing_BW_in_Hz/(Fs/2)) * (samples_per_frame/2));
attack_decay_number_of_samples = attack_time*Fs;
attack_decay_number_of_blocks = 1 + round( attack_decay_number_of_samples / (samples_per_frame/2) );
attack_decay_per_block = 10^( noise_attenuation_in_dB/(20*attack_decay_number_of_blocks) );
release_decay_number_of_samples = release_time*Fs;
release_decay_number_of_blocks = 1 + round( release_decay_number_of_samples / (samples_per_frame/2) );
release_decay_per_block = 10^( noise_attenuation_in_dB/(20*release_decay_number_of_blocks) );
spectral_minimum_search_number_of_blocks = round( spectral_minimum_search_time * Fs / (samples_per_frame/2) );
spectral_minimum_search_number_of_blocks = max( 2 , spectral_minimum_search_number_of_blocks );
gains_history_number_of_blocks = attack_decay_number_of_blocks + release_decay_number_of_blocks - 1; 
gains_history_number_of_blocks = max(gains_history_number_of_blocks,spectral_minimum_search_number_of_blocks);

%Search window history indices:
center_of_spectral_minimum_history_blocks = release_decay_number_of_blocks;
start_of_spectral_minimum_history_blocks = center_of_spectral_minimum_history_blocks - floor(spectral_minimum_search_number_of_blocks/2);
stop_of_spectral_minimum_history_blocks = center_of_spectral_minimum_history_blocks + ceil(spectral_minimum_search_number_of_blocks/2);

%Initialize Matrices containing spectrums and gains history:
spectrums_history = zeros(FFT_size_signal_filter,gains_history_number_of_blocks);
ffts_history = zeros(FFT_size_signal_filter,gains_history_number_of_blocks);
gains_history = ones(FFT_size_signal_filter,gains_history_number_of_blocks) * noise_attenuation_factor;

%Get initial noise-only number of initial seconds:
number_of_initial_seconds_containing_only_noise = 3;
number_of_initial_samples_containing_only_noise = round(Fs*number_of_initial_seconds_containing_only_noise);

%Initialize needed variables:
global_spectrum_max = zeros(FFT_size_signal_filter,1);
initial_noise_data_spectrum_matrix_running_local_min = zeros(FFT_size_signal_filter,spectral_minimum_search_number_of_blocks);
first_and_second_greatest_spectrums = zeros(FFT_size_signal_filter,2);
last_phase_previous = 0;
noise_gate_threshold_vec = zeros(FFT_size_signal_filter,1);
current_sub_frame = zeros(samples_per_frame,1);
current_max_sample = 0;
last_cumsum = 0;
noise_attenuation_constant_vec = noise_attenuation_factor*ones(FFT_size_signal_filter,1);
current_t_vec = (0:samples_per_frame_demodulation-1)'/Fs-(samples_per_frame_demodulation/Fs);
phase_signal_difference = zeros(samples_per_frame_demodulation,1);

%Initialize audacity algorithm indices and indices vecs:
current_fft_update_index = 0;
current_final_fft_index = 1;
current_search_window_indices = flip([4;5;6]);

%FFT & IFFT objects:
FFT_object = dsp.FFT;
IFFT_object = dsp.IFFT;

%audio player object:
audio_player_object = dsp.AudioPlayer;
audio_player_object.SampleRate=44100;
audio_player_object.QueueDuration = 5;

%Get detector frequency calibration:
load('detector_frequency_calibration');
x_frequency = f_vec; 
y_frequency = fft_sum_smoothed;
x_calibration_large_frame = fft_get_frequency_vec(samples_per_frame_demodulation,Fs,0);
x_calibration_sub_frame = fft_get_frequency_vec(samples_per_frame,Fs,0);
y_calibration_sub_frame = interp1(x_frequency,y_frequency,x_calibration_sub_frame); 
y_calibration_sub_frame = fftshift(y_calibration_sub_frame);
y_calibration_large_frame = interp1(x_frequency,y_frequency,x_calibration_large_frame); 
y_calibration_large_frame = fftshift(y_calibration_large_frame);

%Build final hilbert+carrier_filter+y_calibration:
hilbert_filter_order = 256;
hilbert_filter = design(fdesign.hilbert(hilbert_filter_order,0.03));
hilbert_plus_carrier_filter = dfilt.cascade(hilbert_filter,carrier_filter);
hilbert_plus_carrier_filter_fft = hilbert_plus_carrier_filter.freqz(FFT_size_carrier_filter);

%Build final integrator+signal_filter+detector_calibration(+potential equalizer) FFT:
integrator_filter = dfilt.df1(1,[1,-0.999999999999999]);
integrator_plus_signal_filter = dfilt.cascade(integrator_filter,signal_filter);
integrator_plus_signal_filter_fft = integrator_plus_signal_filter.freqz(FFT_size_signal_filter);
integrator_plus_signal_filter_group_delay = round(mean(grpdelay(integrator_plus_signal_filter)));
signal_filter_group_delay = round(mean(grpdelay(signal_filter)));
signal_filter_fft = fft(signal_filter.Numerator,FFT_size_signal_filter);
signal_filter_fft = conj(signal_filter_fft');

%Jump to start_second:
start_second = 5; 
stop_second = start_second+1000;
stat_second = max(start_second,0);
stop_second = min(stop_second,number_of_seconds_in_file);
number_of_frames_to_read = floor((stop_second-start_second)*Fs/samples_per_frame_demodulation);
fseek(fid,8*ceil(start_second*Fs),-1);

frame_counter=1;
flag_skip_initial_delay = 1;
% profile on;
flag_show_large_frame_tic_toc=0;
while frame_counter<number_of_frames_to_read
    tic
 
    %Read current sub-frame and use signal_buffer_object to get overlapping frames:
    current_frame = fread(fid,samples_per_frame_demodulation,'double');
    
    %filter carrier:
    flag_filter_carrier = 1;
    if flag_filter_carrier == 1
        filtered_carrier = step(carrier_filter_object,current_frame);
    else
        filtered_carrier = current_frame;
    end
    
    %extract analytic signal:
    analytic_signal = step(analytic_signal_object,filtered_carrier);
    
    %do FM demodulation (and fill phase_signal_difference to full length):
    phase_signal_difference(2:end) = angle(analytic_signal(2:end).*conj(analytic_signal(1:end-1)));
    first_phase_current = (analytic_signal(1));
    phase_signal_difference(1) = angle(first_phase_current.*conj(last_phase_previous));
    last_phase_previous = (analytic_signal(end));
    
    %throw DC away because i'm not using dan's method anymore so this is needed:
    phase_signal_difference = phase_signal_difference-mean(phase_signal_difference);
 
    %Set thresholds and Build masks:
    click_threshold = 0.0008; %silence where analytic signal amplitude is below this threshold
    indices_to_disregard_click = find( abs(analytic_signal) < click_threshold );  
    
    %Remove Clicks options: 
    flag_use_only_found_only_additional_or_both = 3;
    additional_indices_to_interpolate_over = indices_to_disregard_click;
    number_of_indices_around_spikes_to_delete_as_well = 0;
    flag_use_my_spline_or_matlabs_or_binary_or_do_nothing = 2;
     
    %detect clicks clicks: 
    flag_use_despiking_or_nothing = 1; 
    universal_criterion_multiple=0.9;
    if flag_use_despiking_or_nothing == 1 
        [phase_signal_after_mask,indices_containing_spikes,~] = despike_SOL_fast_with_logical_masks(phase_signal_difference, flag_use_only_found_only_additional_or_both, additional_indices_to_interpolate_over, number_of_indices_around_spikes_to_delete_as_well, flag_use_my_spline_or_matlabs_or_binary_or_do_nothing,universal_criterion_multiple); 
    end     
    
    %Smooth signal from spikes
    binary_mask = ones(size(phase_signal_difference));
    binary_mask(indices_containing_spikes)=0;
    raw_signal_weights = (abs(analytic_signal).^1).*binary_mask;
    derivative_signal_weight_coefficient = 4;
    differentiation_order = 2;
    [phase_signal_after_mask] = smooth_lsq_minimum_iterations_with_preweights(phase_signal_difference,raw_signal_weights,derivative_signal_weight_coefficient,differentiation_order);
    
    %get sub-framed & buffered signal matrix and window them:
    phase_signal_matrix = buffer(phase_signal_after_mask,samples_per_frame,overlap_samples_per_frame);
    phase_signal_matrix = bsxfun(@times,phase_signal_matrix,hanning_window);
    
    %calculate ffts and spectrums with number of elements larger then original size for later overlap-add:
    phase_signal_fft_matrix = fft(phase_signal_matrix,FFT_size_signal_filter);
    phase_signal_spectrum_matrix = abs(phase_signal_fft_matrix).^2;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %START NOISE REMOVAL PROCEDURE:
    current_index_to_update_current_fft = 1;
    start_index = 1;
    stop_index = overlap_samples_per_frame;
    sub_frame_counter = 1;
    flag_show_sub_frame_tic_toc = 0;
    while sub_frame_counter < global_to_sub_frame_size_ratio*2
        
        %disregard first few sub frames because of system objects delay:
        if flag_skip_initial_delay==1
            %zero flag to not enter condition again:
            flag_skip_initial_delay = 0;
            
            %skip sub_frame_counter*overlap_samples_per_frame into the signal:
            initial_sub_frame_skip = 5;
            sub_frame_counter = initial_sub_frame_skip;
            %shift current positions in accordance with us initializing current_sub_frame:
            current_max_sample = overlap_samples_per_frame*2;
            current_max_sample = current_max_sample + overlap_samples_per_frame;
        else
            %shift current_sub_frame memory:
            current_max_sample = current_max_sample + overlap_samples_per_frame;
            sub_frame_counter = sub_frame_counter+1;
        end
          
        %Get current frame fft, magnitude, phase and spectrum: 
        current_sub_frame = phase_signal_matrix(:,sub_frame_counter);
        current_sub_frame_fft = phase_signal_fft_matrix(:,sub_frame_counter);
        current_sub_frame_spectrum = phase_signal_spectrum_matrix(:,sub_frame_counter);

        %Update current frame update index and search window indices:
        current_fft_update_index = mod(current_fft_update_index,gains_history_number_of_blocks) + 1;
        current_final_fft_index = mod(current_final_fft_index,gains_history_number_of_blocks) + 1;
        if frame_counter==1 && current_fft_update_index>=start_of_spectral_minimum_history_blocks
            current_search_window_indices = mod(current_search_window_indices,gains_history_number_of_blocks) + 1;
        end
        
        %Update histories with current fft, spectrum and gain factor:
        ffts_history(:,current_fft_update_index) = current_sub_frame_fft;
        spectrums_history(:,current_fft_update_index) = current_sub_frame_spectrum;
        gains_history(:,current_fft_update_index) = noise_attenuation_factor;
        
        %Get current search window spectrums
        current_search_window_spectrums = spectrums_history(:,current_search_window_indices);
        
        %Sort current search window spectrums in order to get second largest:
        current_sorted_search_window_spectrums = sort(current_search_window_spectrums,2);
        
        %Calculate current search window spectrums meanto build noise gates:
        current_mean_search_window_spectrums = mean(current_search_window_spectrums,2);
        
        %Either:
        %(1). Update Noise Profile:
        noise_profile_flag = 2;
        if current_max_sample < number_of_initial_samples_containing_only_noise
            if noise_profile_flag==1
                %average power spectrum:
                noise_gate_threshold_vec = (noise_gate_threshold_vec*(frame_counter-1) + current_sub_frame_spectrum) / frame_counter;
            elseif noise_profile_flag==2
                %max of min:
                initial_noise_data_spectrum_matrix_running_local_min(:,1) = current_sub_frame_spectrum;
                initial_noise_data_spectrum_matrix_running_local_min = circshift(initial_noise_data_spectrum_matrix_running_local_min,2);
                initial_noise_data_spectrum_running_max_of_min = min(initial_noise_data_spectrum_matrix_running_local_min,[],2);
                
                %find global max of running max of min
                global_spectrum_max = max( global_spectrum_max , initial_noise_data_spectrum_running_max_of_min );
                
                %Assign max of min to noise gate thresholds:
                noise_gate_threshold_vec = global_spectrum_max;
            elseif noise_profile_flag==3
                %get second greatest:
                indices_to_update_first_greatest_logical_mask = (current_sub_frame_spectrum>first_and_second_greatest_spectrums(:,1));
                indices_to_update_second_greatest_logical_mask = (current_sub_frame_spectrum<first_and_second_greatest_spectrums(:,1) & current_sub_frame_spectrum>first_and_second_greatest_spectrums(:,2));
                first_and_second_greatest_spectrums(indices_to_update_first_greatest_logical_mask,1) = current_sub_frame_spectrum(indices_to_update_first_greatest_logical_mask);
                first_and_second_greatest_spectrums(indices_to_update_second_greatest_logical_mask,2) = current_sub_frame_spectrum(indices_to_update_second_greatest_logical_mask);
                noise_gate_threshold_vec = first_and_second_greatest_spectrums(:,2);
            end
        end
        
        
        %(2). Reduce Noise:
        if current_max_sample > number_of_initial_samples_containing_only_noise
            %raise the gain for elements in the center of the sliding history:
            %(*) The assumption is that at first all indices are BELOW noise gate,
            %and it is only when an index is found to be above the gate (according
            %to our criteria) that we change something and raise the center index
            %to 1, which then decays according to our rules.
            %(*) If current history is found not to be above noise gate then it is
            %passed through WITHOUT CHANGE, and so it can be effected by both
            %forward frames and backward frames WHICH CAN ONLY RAISE(!) IT
            noise_classification_flag_counter=1;
            if noise_classification_flag_counter==1
                %check if second greatest in search window is above noise threshold
                indices_above_noise_gate = find(current_sorted_search_window_spectrums(:,end-1) > sensitivity_factor * noise_gate_threshold_vec);
                
            elseif noise_classification_flag_counter==2
                %check if current window mean is above noise threshold
                indices_above_noise_gate = find(current_mean_search_window_spectrums > sensitivity_factor * noise_gate_threshold_vec);
            end
            if ~isempty(indices_above_noise_gate)
                gains_history(indices_above_noise_gate,center_of_spectral_minimum_history_blocks) = 1;
            end
            
            %DECAY THE GAIN IN BOTH DIRECTIONS:
            %(*) if the gain before or after is lower then what one gets when we decay the current center gain
            % then RAISE it according to decay rate from center gain:
            %HOLD (backward in time):
            for history_frame_counter = center_of_spectral_minimum_history_blocks+1 : 1 : gains_history_number_of_blocks
                gains_history(:,history_frame_counter) = max( gains_history(:,history_frame_counter-1)*attack_decay_per_block, max(noise_attenuation_constant_vec,gains_history(:,history_frame_counter)) );
            end
            %RELEASE (forward in time):
            for history_frame_counter = center_of_spectral_minimum_history_blocks-1 : -1 : 1
                gains_history(:,history_frame_counter) = max( gains_history(:,history_frame_counter+1)*release_decay_per_block, max(noise_attenuation_constant_vec,gains_history(:,history_frame_counter)) );
            end
            
        end
        
        %Get current end fft and gain:
        current_gain_function = gains_history(:,current_final_fft_index);
        current_fft = (ffts_history(:,current_final_fft_index));
        
        %Smooth fft log-wise:
        flag_use_linear_or_log_averaging = 2;
        if number_of_frequency_bins>1
            if flag_use_linear_or_log_averaging==1
                current_gain_function = conv(current_gain_function,ones(1,number_of_frequency_bins),'same')/number_of_frequency_bins;
            elseif flag_use_linear_or_log_averaging==2
                current_log_gain_function = log(current_gain_function);
                current_log_gain_function_averaged = conv(current_gain_function,ones(1,number_of_frequency_bins),'same')/number_of_frequency_bins;
                current_gain_function_final = exp(current_log_gain_function_averaged);
            end
        else
            current_gain_function_final = current_gain_function;
        end
         
        %Multiply current fft by gain function (SHOULD I ADD DEWINDOWING???):
        fft_after_gain_function = current_fft.*current_gain_function_final.*(signal_filter_fft) + eps*1i;
        enhanced_frame_current_expanded = real(step(IFFT_object,fft_after_gain_function));
        enhanced_frame_current = enhanced_frame_current_expanded(signal_filter_group_delay+1:signal_filter_group_delay+samples_per_frame);
        enhanced_frame_current = enhanced_frame_current.*deframing_window;
        
        %Overlap-Add to get final signal: 
        first_overlapping_part_of_enhanced_signal = enhanced_frame_current(1:overlap_samples_per_frame) + enhanced_speech_overlapped_samples_from_previous_frame;
        
        %Remember overlapping part for next overlap-add:
        enhanced_speech_overlapped_samples_from_previous_frame = enhanced_frame_current( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame);
        
        %Integrate final signal and filter it:
        final_signal = cumsum(first_overlapping_part_of_enhanced_signal) + last_cumsum;
        last_cumsum = final_signal(end);
        
        %Sound final signal:
        flag_sound_demodulation=1;
        if flag_sound_demodulation==1
            step(audio_player_object,[final_signal(:),final_signal(:)]);
        end
        
        %Save to .wav file if wanted:
        flag_save_to_wav = 0; 
        if flag_save_to_wav==1
            step(audio_file_writer_demodulated,final_signal(:)/3);
        end
        
    end %sub-frame loop
%     profile off;
%     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    toc
    frame_counter=frame_counter+1;
end
% profview

try
    fclose(fid);
    fclose('all');
    release(audio_player_object);
    release(audio_file_writer_demodulated);
    release(audio_file_reader);
    release(analytic_signal_object);
catch 
end















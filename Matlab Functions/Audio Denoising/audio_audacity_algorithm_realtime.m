%audio audacity algorithm realtime:

%Input variables:
input_file_name = 'shirt_2mm_ver_200m_audioFM final demodulated audio150-3000[Hz]';
output_file_name = 'blabla.wav';

%Read input file:
[input_signal, Fs, bits] = wavread( input_file_name);
input_signal = make_column(input_signal(2500:end));

%Audio parameters:
% frame_size_in_seconds = 0.02;   
% RBW = 1/frame_size_in_seconds;
% samples_per_frame = make_even(floor(Fs*frame_size_in_seconds),1);
samples_per_frame = 2048;
overlap_samples_per_frame = (samples_per_frame * 1/2);	
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
hanning_window = make_column(hanning(samples_per_frame));
normalization_factor = ( hanning_window'* hanning_window)/ samples_per_frame;
FFT_size = 2^(nextpow2(samples_per_frame));

%Input Constant Parameters:
noise_attenuation_in_dB = -0;
sensitivity_in_dB = 10; 
frequency_smoothing_BW_in_Hz = 0;
number_of_frequency_bins = round(frequency_smoothing_BW_in_Hz/(1/(samples_per_frame/Fs)));
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

%Initialize Matrices containing spectrums and gains history:
spectrums_history = zeros(gains_history_number_of_blocks,FFT_size);
ffts_history = zeros(gains_history_number_of_blocks,FFT_size);
gains_history = ones(gains_history_number_of_blocks,FFT_size) * noise_attenuation_factor;


%Get initial noise-only data samples and get noise thresholds:
number_of_initial_seconds_containing_only_noise = 0.5;
number_of_initial_samples_containing_only_noise = round(Fs*number_of_initial_seconds_containing_only_noise);
initial_samples_containing_only_noise = input_signal(1:number_of_initial_samples_containing_only_noise);
initial_noise_data_matrix = buffer(initial_samples_containing_only_noise,samples_per_frame,overlap_samples_per_frame);
initial_noise_data_matrix = bsxfun(@times,initial_noise_data_matrix,hanning_window);
initial_noise_data_spectrum_matrix = abs(fft(initial_noise_data_matrix,FFT_size)).^2;


%Search window history indices:
center_of_spectral_minimum_history_blocks = release_decay_number_of_blocks;
start_of_spectral_minimum_history_blocks = center_of_spectral_minimum_history_blocks - floor(spectral_minimum_search_number_of_blocks/2);
stop_of_spectral_minimum_history_blocks = center_of_spectral_minimum_history_blocks + ceil(spectral_minimum_search_number_of_blocks/2);

%Get only noisy speech after initial noise-only samples:
% noisy_speech = input_signal(number_of_initial_samples_containing_only_noise : end);
noisy_speech = input_signal;
noisy_speech_number_of_samples = length(noisy_speech);
noisy_speech_number_of_frames = fix( (noisy_speech_number_of_samples-overlap_samples_per_frame) / non_overlapping_samples_per_frame);
initial_indices_of_frames_in_noisy_signal = 1 + (0:(noisy_speech_number_of_frames-1))*non_overlapping_samples_per_frame;
number_of_samples_resulting_from_buffering_input_signal = samples_per_frame + initial_indices_of_frames_in_noisy_signal(noisy_speech_number_of_frames) - 1;
if noisy_speech_number_of_samples < number_of_samples_resulting_from_buffering_input_signal
   %Zero pad signal to equate number of samples of buffered and original signal if necessary:
   noisy_speech(noisy_speech_number_of_samples+1 : samples_per_frame+initial_indices_of_frames_in_noisy_signal(noisy_speech_number_of_frames)-1) = 0;  
end

%Initialize helper variable which contains the overlapping samples of the previous frame:
enhanced_speech_overlapped_samples_from_previous_frame = zeros(overlap_samples_per_frame,1);


%Signal Source Object:
signal_source_object = dsp.SignalSource;
signal_source_object.Signal = input_signal;
signal_source_object.SamplesPerFrame = overlap_samples_per_frame;
%Signal Buffer Object:
signal_buffer_object = dsp.Buffer;
signal_buffer_object.Length = samples_per_frame;
signal_buffer_object.OverlapLength = overlap_samples_per_frame;

%FFT & IFFT objects:
FFT_object = dsp.FFT;
IFFT_object = dsp.IFFT;

%Audio Plyaer Object:
audio_player_object = dsp.AudioPlayer;
audio_player_object.SampleRate = Fs;
audio_player_object.QueueDuration = 5;

%Initialize needed variables:
first_and_seoncd_greatest_spectrums = zeros(2,samples_per_frame);
global_spectrum_max = zeros(1,samples_per_frame);
initial_noise_data_spectrum_matrix_running_local_min = zeros(spectral_minimum_search_number_of_blocks,samples_per_frame);
first_and_second_greatest_spectrums = zeros(2,samples_per_frame);
noise_gate_threshold_vec = zeros(1,samples_per_frame);
%Loop over noisy frames and enhance them:
for frame_counter = 1:noisy_speech_number_of_frames
    tic
    %Get start and stop indices of current frame:
    start_index = initial_indices_of_frames_in_noisy_signal(frame_counter);
    stop_index = initial_indices_of_frames_in_noisy_signal(frame_counter) + samples_per_frame - 1;
    
    
    %Current sample count:
    current_max_sample = frame_counter * overlap_samples_per_frame;
    
    %Get noisy speech frame:
    current_frame = step(signal_buffer_object,step(signal_source_object));
    
    %Window current frame:
    current_frame = current_frame.*hanning_window;
    
    %Get current frame fft, magnitude, phase and spectrum:
    current_frame_fft = step(FFT_object,current_frame);
    current_frame_fft_magnitude = abs(current_frame_fft);
    current_frame_fft_phase  = angle(current_frame_fft);
    current_frame_spectrum = current_frame_fft_magnitude.^ 2;
    
    %Update histories:
    ffts_history(1,:) = current_frame_fft;
    spectrums_history(1,:) = current_frame_spectrum;
    gains_history(1,:) = noise_attenuation_factor;
    current_search_window_spectrums = spectrums_history(start_of_spectral_minimum_history_blocks:stop_of_spectral_minimum_history_blocks,:);
    current_sorted_search_window_spectrums = sort(current_search_window_spectrums,1);
    current_mean_search_window_spectrums = mean(current_search_window_spectrums,1);
    
    %Either:
    %(1). Update Noise Profile:
    noise_profile_flag = 3;
    if current_max_sample < number_of_initial_samples_containing_only_noise
        if noise_profile_flag==1
            %average power spectrum:
            noise_gate_threshold_vec = (noise_gate_threshold_vec*(frame_counter-1) + current_frame_spectrum) / frame_counter; 
        elseif noise_profile_flag==2
            %max of min:
            initial_noise_data_spectrum_matrix_running_local_min(1,:) = current_frame_spectrum;
            initial_noise_data_spectrum_matrix_running_local_min = circshift(initial_noise_data_spectrum_matrix_running_local_min,1);
            initial_noise_data_spectrum_running_max_of_min = max(initial_noise_data_spectrum_matrix_running_local_min,[],1);
            
            %find global max of running max of min
            global_spectrum_max = max( global_spectrum_max , initial_noise_data_spectrum_running_max_of_min );
            
            %Assign max of min to noise gate thresholds:
            noise_gate_threshold_vec = global_spectrum_max;
        elseif noise_profile_flag==3
            %get second greatest:
            indices_to_update_first_greatest = find(current_frame_spectrum>first_and_second_greatest_spectrums(1,:)');
            indices_to_update_second_greatest = find(current_frame_spectrum<first_and_second_greatest_spectrums(1,:)' & current_frame_spectrum>first_and_second_greatest_spectrums(2,:)');
            first_and_second_greatest_spectrums(1,indices_to_update_first_greatest) = current_frame_spectrum(indices_to_update_first_greatest);
            first_and_second_greatest_spectrums(2,indices_to_update_second_greatest) = current_frame_spectrum(indices_to_update_second_greatest);
            noise_gate_threshold_vec = first_and_second_greatest_spectrums(1,:);
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
            indices_above_noise_gate = find(current_sorted_search_window_spectrums(round(size(current_sorted_search_window_spectrums,1)/2),:) > sensitivity_factor * noise_gate_threshold_vec);
            
        elseif noise_classification_flag_counter==2
            %check if current window mean is above noise threshold
            indices_above_noise_gate = find(current_mean_search_window_spectrums > sensitivity_factor * noise_gate_threshold_vec');
        end
        if ~isempty(indices_above_noise_gate)
            gains_history(center_of_spectral_minimum_history_blocks, indices_above_noise_gate) = 1;
        end
        
        %DECAY THE GAIN IN BOTH DIRECTIONS:
        %(*) if the gain before or after is lower then what one gets when we decay the current center gain
        % then RAISE it according to decay rate from center gain:
        %HOLD (backward in time):
        for history_frame_counter = center_of_spectral_minimum_history_blocks+1 : 1 : gains_history_number_of_blocks
           gains_history(history_frame_counter,:) = max( gains_history(history_frame_counter-1,:)*attack_decay_per_block, max(noise_attenuation_factor*ones(1,FFT_size),gains_history(history_frame_counter,:)) ); 
        end
        %RELEASE (forward in time):
        for history_frame_counter = center_of_spectral_minimum_history_blocks-1 : -1 : 1
            gains_history(history_frame_counter,:) = max( gains_history(history_frame_counter+1,:)*release_decay_per_block, max(noise_attenuation_factor*ones(1,FFT_size),gains_history(history_frame_counter,:)) );
        end
        
    end
    
    %Shift history windows:
    ffts_history = circshift(ffts_history,[1,0]);
    spectrums_history = circshift(spectrums_history,[1,0]);
    gains_history = circshift(gains_history,[1,0]);
    
    
    %Calculate Enhanced speech frame: 
    current_gain_function = gains_history(end,:);
    current_fft = conj(ffts_history(end,:))';
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
    corrected_fft = current_fft.*current_gain_function_final' + eps*1i;
    enhanced_frame_current = real(step(IFFT_object,corrected_fft(:)));
    
    
    
    %Overlap-Add:
    first_overlapping_part_of_enhanced_signal = enhanced_frame_current(1:overlap_samples_per_frame) + enhanced_speech_overlapped_samples_from_previous_frame;
    second_nonoverlapping_part_of_enhanced_signal = enhanced_frame_current(overlap_samples_per_frame+1 : samples_per_frame);
    current_final_enhanced_speech_frame = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
    final_enhanced_speech(start_index:stop_index) = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
    
    step(audio_player_object,first_overlapping_part_of_enhanced_signal);
   
    %Remember overlapping part for next overlap-add:
    enhanced_speech_overlapped_samples_from_previous_frame = enhanced_frame_current( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame);
    toc
end    
 
% final_enhanced_speech( start_index : start_index + samples_per_frame/2- 1) = enhanced_speech_overlapped_samples_from_previous_frame; 
 
% sound(final_enhanced_speech'*20,Fs);
wavwrite( final_enhanced_speech, Fs, bits, output_file_name);
toc





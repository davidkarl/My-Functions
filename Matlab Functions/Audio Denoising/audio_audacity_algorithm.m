% function audio_audacity_algorithm(input_file_name,output_file_name)

overlap_samples_per_frame_vec = [1/2];
noise_profile_flag_vec = [1,2,3];
noise_classification_flag_vec = [1,2];
number_of_frequency_bins_vec = [1];
sensitivity_vec = [6,10];
noise_attenuation_vec = [-10,-20];

for overlap_samples_per_frame_counter = 1:length(overlap_samples_per_frame_vec)
    overlap_fraction = overlap_samples_per_frame_vec(overlap_samples_per_frame_counter);
    for noise_profile_flag_counter = 1:length(noise_profile_flag_vec)
        for noise_classification_flag_counter = 1:length(noise_classification_flag_vec)
            for number_of_frequency_bins_counter = 1:length(number_of_frequency_bins_vec)
                number_of_frequency_bins = number_of_frequency_bins_vec(number_of_frequency_bins_counter);
                for sensitivity_counter = 1:length(sensitivity_vec)
                    sensitivity_in_dB = sensitivity_vec(sensitivity_counter);
                    for noise_attenuation_counter = 1:length(noise_attenuation_vec)
                        tic
                        noise_attenuation_in_dB = noise_attenuation_vec(noise_attenuation_counter);
%Input variables:
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio170-3500[Hz]lsq smoother with deframing';
output_file_name = strcat( 'overlap fraction=',num2str(overlap_fraction),',noise profile flag=',num2str(noise_profile_flag_counter),...
    ',noise classification flag=',num2str(noise_classification_flag_counter),',number of frequency bins=',num2str(number_of_frequency_bins),...
    ',sensitivity=',num2str(sensitivity_vec(sensitivity_counter)),',noise attenuation=',num2str(noise_attenuation_in_dB),'.wav');

%Read input file:
[input_signal, Fs, bits] = wavread( input_file_name);
input_signal = make_column(input_signal(2500:end));

%Audio parameters:
% frame_size_in_seconds = 0.02;   
% RBW = 1/frame_size_in_seconds;
% samples_per_frame = make_even(floor(Fs*frame_size_in_seconds),1);
samples_per_frame = 2048;
overlap_samples_per_frame = (samples_per_frame * overlap_fraction);	
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
hanning_window = make_column(hanning(samples_per_frame));
normalization_factor = ( hanning_window'* hanning_window)/ samples_per_frame;
FFT_size = 2^(nextpow2(samples_per_frame));

%Input Constant Parameters:
% noise_attenuation_in_dB = -20;
% sensitivity_in_dB = 6; 
% frequency_smoothing_BW_in_Hz = 0;
% number_of_frequency_bins = round(frequency_smoothing_BW_in_Hz/(1/(samples_per_frame/Fs)));
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
% gains_history_number_of_blocks = 2*max(attack_decay_number_of_blocks,release_decay_number_of_blocks) - 1;
% gains_history_number_of_blocks = max( gains_history_number_of_blocks , spectral_minimum_search_number_of_blocks );
gains_history_number_of_blocks = attack_decay_number_of_blocks + release_decay_number_of_blocks - 1; 
gains_history_number_of_blocks = max(gains_history_number_of_blocks,spectral_minimum_search_number_of_blocks);

%Initialize Matrices containing spectrums and gains history:
spectrums_history = zeros(gains_history_number_of_blocks,FFT_size);
ffts_history = zeros(gains_history_number_of_blocks,FFT_size);
gains_history = ones(gains_history_number_of_blocks,FFT_size) * noise_attenuation_factor;


%Get initial noise-only data samples and get noise thresholds:
% noise_profile_flag_counter = 2;
number_of_initial_seconds_containing_only_noise = 0.5;
number_of_initial_samples_containing_only_noise = round(Fs*number_of_initial_seconds_containing_only_noise);
initial_samples_containing_only_noise = input_signal(1:number_of_initial_samples_containing_only_noise);
initial_noise_data_matrix = buffer(initial_samples_containing_only_noise,samples_per_frame,overlap_samples_per_frame);
initial_noise_data_matrix = bsxfun(@times,initial_noise_data_matrix,hanning_window);
initial_noise_data_spectrum_matrix = abs(fft(initial_noise_data_matrix,FFT_size)).^2;
average_noise_spectrum = mean(initial_noise_data_spectrum_matrix,2);
if noise_profile_flag_counter==1
    %average:
    noise_gate_threshold_vec = mean(initial_noise_data_spectrum_matrix,2);
    
elseif noise_profile_flag_counter==2
    %max of min:
    initial_noise_data_spectrum_matrix_running_local_min = zeros(FFT_size,size(initial_noise_data_matrix,2)-spectral_minimum_search_number_of_blocks+1);
    for k=1:size(initial_noise_data_matrix,2)-spectral_minimum_search_number_of_blocks+1
        initial_noise_data_spectrum_matrix_running_local_min(:,k) = min(initial_noise_data_spectrum_matrix(:,k:k+spectral_minimum_search_number_of_blocks-1),[],2);
    end
    noise_gate_threshold_vec = max(initial_noise_data_spectrum_matrix_running_local_min,[],2);
    
elseif noise_profile_flag_counter==3
    %second greatest:
    initial_noise_data_spectrum_matrix_sorted = sort(initial_noise_data_spectrum_matrix,2);
    noise_gate_threshold_vec = initial_noise_data_spectrum_matrix_sorted(:,end-1);
    
end



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


%Loop over noisy frames and enhance them:
for frame_counter = 1:noisy_speech_number_of_frames
%     tic
    %Get start and stop indices of current frame:
    start_index = initial_indices_of_frames_in_noisy_signal(frame_counter);
    stop_index = initial_indices_of_frames_in_noisy_signal(frame_counter) + samples_per_frame - 1;
    
    %Get noisy speech frame:
    current_frame = noisy_speech(start_index : stop_index);
    
    %Window current frame:
    current_frame = current_frame.*hanning_window;
    
    %Get current frame fft, magnitude, phase and spectrum:
    current_frame_fft = fft(current_frame,FFT_size);
    current_frame_fft_magnitude = abs(current_frame_fft);
    current_frame_fft_phase  = angle(current_frame_fft);
    current_frame_spectrum = current_frame_fft_magnitude.^ 2;
    ffts_history(1,:) = current_frame_fft;
    spectrums_history(1,:) = current_frame_spectrum;
    gains_history(1,:) = noise_attenuation_factor;
    
    %Search for spectral minimum over search blocks to compare to noise gate: 
    %
    center_of_spectral_minimum_history_blocks = release_decay_number_of_blocks;
    start_of_spectral_minimum_history_blocks = center_of_spectral_minimum_history_blocks - floor(spectral_minimum_search_number_of_blocks/2);
    stop_of_spectral_minimum_history_blocks = center_of_spectral_minimum_history_blocks + ceil(spectral_minimum_search_number_of_blocks/2);
    
    %
    %raise the gain for elements in the center of the sliding history:
    %(*) The assumption is that at first all indices are BELOW noise gate,
    %and it is only when an index is found to be above the gate (according
    %to our criteria) that we change something and raise the center index
    %to 1, which then decays according to our rules. 
    %(*) If current history is found not to be above noise gate then it is
    %passed through WITHOUT CHANGE, and so it can be effected by both
    %forward frames and backward frames WHICH CAN ONLY RAISE(!) IT
    current_search_window_spectrums = spectrums_history(start_of_spectral_minimum_history_blocks:stop_of_spectral_minimum_history_blocks,:);
    current_sorted_search_window_spectrums = sort(current_search_window_spectrums,1);
    current_mean_search_window_spectrums = mean(current_search_window_spectrums,1);
    
%     noise_classification_flag_counter=1;
    if noise_classification_flag_counter==1
       %check if second greatest in search window is above noise threshold 
       indices_above_noise_gate = find(current_sorted_search_window_spectrums(round(size(current_sorted_search_window_spectrums,1)/2),:) > sensitivity_factor * noise_gate_threshold_vec');
       
    elseif noise_classification_flag_counter==2
       %check if current window mean is above noise threshold
       indices_above_noise_gate = find(current_mean_search_window_spectrums > sensitivity_factor * noise_gate_threshold_vec');
    end
    if ~isempty(indices_above_noise_gate)
       gains_history(center_of_spectral_minimum_history_blocks, indices_above_noise_gate) = 1;
    end
     
    %
    %decay the gain in both directions:
    %(*) if the gain before or after is lower then what one gets when we decay the current center gain
    % then RAISE it according to decay rate from center gain:
    %HOLD (backward in time):
    for spectrum_counter=1:FFT_size
        for history_frame_counter = center_of_spectral_minimum_history_blocks+1 : 1 : gains_history_number_of_blocks
            maxima = max(noise_attenuation_factor,gains_history(history_frame_counter-1,spectrum_counter)*release_decay_per_block);
            if gains_history(history_frame_counter,spectrum_counter) < maxima
               gains_history(history_frame_counter,spectrum_counter) = maxima;
            else
               break; 
            end
            
        end
    end
    %RELEASE (forward in time):
    for history_frame_counter = center_of_spectral_minimum_history_blocks-1 : -1 : 1
        gains_history(history_frame_counter,:) = max( gains_history(history_frame_counter+1,:)*release_decay_per_block, max(noise_attenuation_factor*ones(1,FFT_size),gains_history(history_frame_counter,:)) );
    end
     
    %Shift history windows:
    ffts_history = circshift(ffts_history,[1,0]);
    spectrums_history = circshift(spectrums_history,[1,0]);
    gains_history = circshift(gains_history,[1,0]);
    
    
    %Calculate Enhanced speech frame: 
    current_gain_function = gains_history(end,:);
    current_fft = conj(ffts_history(end,:))';
%     number_of_frequency_bins = 2;
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
    corrected_fft = current_fft.*current_gain_function_final';
    enhanced_frame_current = real(ifft( corrected_fft, FFT_size));
    
    %Overlap-Add:
    first_overlapping_part_of_enhanced_signal = enhanced_frame_current(1:overlap_samples_per_frame) + enhanced_speech_overlapped_samples_from_previous_frame;
    second_nonoverlapping_part_of_enhanced_signal = enhanced_frame_current(overlap_samples_per_frame+1 : samples_per_frame);
    current_final_enhanced_speech_frame = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
    final_enhanced_speech(start_index:stop_index) = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
    
    %Remember overlapping part for next overlap-add:
    enhanced_speech_overlapped_samples_from_previous_frame = enhanced_frame_current( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame);
    
%     toc
end    
 
% final_enhanced_speech( start_index : start_index + samples_per_frame/2- 1) = enhanced_speech_overlapped_samples_from_previous_frame; 
 
% sound(final_enhanced_speech'*20,Fs);
wavwrite( final_enhanced_speech, Fs, bits, output_file_name);
toc
                    end
                end
            end
        end
    end
end


 
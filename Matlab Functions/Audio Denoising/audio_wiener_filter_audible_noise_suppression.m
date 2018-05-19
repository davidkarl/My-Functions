function audio_wiener_filter_audible_noise_suppression(input_file_name,output_file_name)

%Input variables:
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio170-3500[Hz]lsq smoother with deframing';
% input_file_name = 'shirt_2mm_ver_200m_audioFM final demodulated audio150-3000[Hz]';
output_file_name = 'rodena_speaking_spectral_subtraction.wav';

%Read input file:
[input_signal, Fs, bits] = wavread( input_file_name);
input_signal = make_column(input_signal);
input_signal = add_noise_of_certain_SNR(input_signal,0,1,0);

%Audio parameters:
frame_size_in_seconds = 0.02;   
samples_per_frame = make_even(floor(Fs*frame_size_in_seconds),1);
overlap_samples_per_frame = samples_per_frame/ 2;	
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
hamming_window = make_column(hamming(samples_per_frame));
normalization_factor = ( hamming_window'* hamming_window)/ samples_per_frame;
FFT_size = samples_per_frame;

%Get initial noise-only data samples and create a windowed matrix out of them:
number_of_initial_seconds_containing_only_noise = 0.12;
number_of_initial_samples_containing_only_noise = round(Fs*number_of_initial_seconds_containing_only_noise);
initial_samples_containing_only_noise = input_signal(1:number_of_initial_samples_containing_only_noise);
initial_noise_data_matrix = buffer(initial_samples_containing_only_noise,samples_per_frame,overlap_samples_per_frame);
initial_noise_data_matrix = bsxfun(@times,initial_noise_data_matrix,hamming_window);
average_noise_power_spectrum = mean( abs(fft(initial_noise_data_matrix,FFT_size)).^2 , 2);

%Get only noisy speech after initial noise-only samples:
noisy_speech = input_signal(number_of_initial_samples_containing_only_noise+1 : end);
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

%Estimate noise spectrum only in the critical bands:
average_noise_power_spectrum_only_in_critical_bands_one_sided = zeros(samples_per_frame/2+1,1);
[critical_bands_indices_in_linear_frequency_vec] = fft_get_critical_band_indices_in_linear_frequency_vec(Fs,samples_per_frame,16,samples_per_frame/2);
for i = 1:length(critical_bands_indices_in_linear_frequency_vec)
    average_noise_power_spectrum_only_in_critical_bands_one_sided(critical_bands_indices_in_linear_frequency_vec{i}) = ones(size(critical_bands_indices_in_linear_frequency_vec{i},2),1)*mean(average_noise_power_spectrum(critical_bands_indices_in_linear_frequency_vec{i}));
end

%Set parameter values
smoothing_factor_in_noise_spectrum_update = 0.98;
smoothing_factor_in_apriori_SNR_update = 0.98;
VAD_threshold = 0.15; 
spectral_substraction_floor = 0.002;
number_of_iterations_in_evaluating_masking_thresholds = 2;

%Loop over noisy frames and enhance them:
for frame_counter = 1:noisy_speech_number_of_frames
    tic
    %Get start and stop indices of current frame:
    start_index = initial_indices_of_frames_in_noisy_signal(frame_counter);
    stop_index = initial_indices_of_frames_in_noisy_signal(frame_counter) + samples_per_frame - 1;
    
    %Get noisy speech frame:
    current_frame = noisy_speech(start_index : stop_index);
    
    %Window current frame:
    current_frame = current_frame .* hamming_window;
    
    %Get current frame fft, magnitude, phase and spectrum:
    current_frame_fft = fft(current_frame,FFT_size);
    current_frame_fft_magnitude = abs(current_frame_fft);
    current_frame_fft_phase  = angle(current_frame_fft);
    current_frame_power_spectrum = current_frame_fft_magnitude.^ 2;
    
    %aposteriori SNR estimate:
    aposteriori_SNR_estimate_per_frequency_current = current_frame_power_spectrum ./ average_noise_power_spectrum;
    aposteriori_prime_current = max(aposteriori_SNR_estimate_per_frequency_current-1,0); 
    
    %apriori SNR estimate:
    if frame_counter==1
       apriori_SNR_estimate_per_frequency_current_using_wiener = 1; 
    else
       apriori_SNR_estimate_per_frequency_current_using_wiener = (gain_function_previous.^2) .* aposteriori_SNR_estimate_per_frequency_previous;  
    end
    
    %Smooth apriori SNR estimate and calculate respective aposteriori smoothed SNR estimate:
    apriori_SNR_estimate_smoothed = smoothing_factor_in_apriori_SNR_update * apriori_SNR_estimate_per_frequency_current_using_wiener ...
                                  + (1-smoothing_factor_in_apriori_SNR_update) * aposteriori_prime_current;
    aposteriori_SNR_estimate_smoothed = apriori_SNR_estimate_smoothed + 1;
    
    %VAD decide whether speech is present and if not then update noise spectrum:
    log_likelihood_per_frequency = (aposteriori_SNR_estimate_per_frequency_current./aposteriori_SNR_estimate_smoothed).*apriori_SNR_estimate_smoothed - log(aposteriori_SNR_estimate_smoothed);
    vad_decision(frame_counter) = sum(log_likelihood_per_frequency) / samples_per_frame;   
    if (vad_decision(frame_counter) < VAD_threshold) 
        average_noise_power_spectrum = smoothing_factor_in_noise_spectrum_update*average_noise_power_spectrum + (1- smoothing_factor_in_noise_spectrum_update)*current_frame_power_spectrum;
        vad_over_time( start_index : stop_index ) = 0;
    else
        vad_over_time( start_index : stop_index ) = 1;
    end
    
    %Get noise power spectrum only in the critical bands:
    for i = 1:length(critical_bands_indices_in_linear_frequency_vec)
        average_noise_power_spectrum_only_in_critical_bands_one_sided(critical_bands_indices_in_linear_frequency_vec{i})=...
            ones(size(critical_bands_indices_in_linear_frequency_vec{i},2),1) * mean(average_noise_power_spectrum(critical_bands_indices_in_linear_frequency_vec{i}));
    end
    
    %Floor cleaned speech power spectrum to minimum spectral floor: 
    current_frame_power_spectrum_floored = max(current_frame_power_spectrum-average_noise_power_spectrum,spectral_substraction_floor*current_frame_power_spectrum);  
    
    %Get one sided power spectrum to insert into the following "mask" function:
    current_frame_power_spectrum_floored_one_sided = current_frame_power_spectrum_floored(1:samples_per_frame/2+1);

    %Get initial noise masking threshold estimate and initialize power spectrum estimate for wiener filter estimate:
    current_iteration_power_spectrum_using_wiener_filter = current_frame_power_spectrum_floored_one_sided;
    noise_masking_threshold = audio_get_masking_threshold(current_iteration_power_spectrum_using_wiener_filter,samples_per_frame,Fs,16);
    
    %Estimate masking thresholds iteratively (similar to the iterative wiener filter):
    for j=1:number_of_iterations_in_evaluating_masking_thresholds
        alpha_parameter_in_wiener_filter_using_masking_thresholds = average_noise_power_spectrum_only_in_critical_bands_one_sided + (average_noise_power_spectrum_only_in_critical_bands_one_sided.^2)./noise_masking_threshold;
        current_iteration_wiener_filter = (current_iteration_power_spectrum_using_wiener_filter) ./ (alpha_parameter_in_wiener_filter_using_masking_thresholds+current_iteration_power_spectrum_using_wiener_filter);
        current_iteration_power_spectrum_using_wiener_filter = current_iteration_wiener_filter .* current_iteration_power_spectrum_using_wiener_filter;
        noise_masking_threshold = audio_get_masking_threshold(current_iteration_power_spectrum_using_wiener_filter,samples_per_frame,Fs,16);
    end

    %Get final noise masking threshold estimate and final wiener filter estimate:
    alpha_parameter_in_wiener_filter_using_masking_thresholds = average_noise_power_spectrum_only_in_critical_bands_one_sided + (average_noise_power_spectrum_only_in_critical_bands_one_sided.^2)./noise_masking_threshold;  
    final_wiener_filter_estimation = (current_frame_power_spectrum_floored_one_sided./(alpha_parameter_in_wiener_filter_using_masking_thresholds+current_frame_power_spectrum_floored_one_sided));
    gain_function_current = [final_wiener_filter_estimation(1:samples_per_frame/2+1); flipud(final_wiener_filter_estimation(2:samples_per_frame/2))];
    
    %Calculate current frame weighted fft using calculated gain function on current frame fft:
    current_frame_weighted_fft_after_gain_function = gain_function_current .* current_frame_fft;
    
    %Calculate current enhanced speech frame:
    enhanced_frame_current = real(ifft(current_frame_weighted_fft_after_gain_function));
    
    %Overlap-Add:
    first_overlapping_part_of_enhanced_signal = enhanced_frame_current(1:overlap_samples_per_frame) + enhanced_speech_overlapped_samples_from_previous_frame;
    second_nonoverlapping_part_of_enhanced_signal = enhanced_frame_current(overlap_samples_per_frame+1 : samples_per_frame);
    current_final_enhanced_speech_frame = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
    final_enhanced_speech(start_index:stop_index) = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
    
    %Remember overlapping part for next overlap-add:
    enhanced_speech_overlapped_samples_from_previous_frame = enhanced_frame_current( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame);
    
    %Remember current gain function for later apriori SNR estimation: 
    gain_function_previous = gain_function_current; 
    aposteriori_SNR_estimate_per_frequency_previous = aposteriori_SNR_estimate_per_frequency_current;
    toc
end
      
%Write final enhanced speech to .wav file:
sound(final_enhanced_speech,Fs);
wavwrite( final_enhanced_speech, Fs, bits, output_file_name);



 
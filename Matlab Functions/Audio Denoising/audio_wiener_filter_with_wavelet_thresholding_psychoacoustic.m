function output_file_name= audio_wiener_filter_with_wavelet_thresholding_psychoacoustic( input_file_name, output_file_name) 

%Input variables:
input_file_name = '900m, reference loudness=69dBspl, conversation, laser power=1 , Fs=4935Hz, channel2.wav';
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio170-3500[Hz]lsq smoother with deframing';
output_file_name = 'rodena_speaking_spectral_subtraction.wav';

%Read input file:
[input_signal, Fs, bits] = wavread(input_file_name);
input_signal = make_column(input_signal);
% input_signal = add_noise_of_certain_SNR(input_signal,-10,1,0);

%Audio parameters:
frame_size_in_seconds = 0.02;   
overlap_fraction_between_frames = 0.5;
samples_per_frame = make_even(floor(Fs*frame_size_in_seconds),1);
overlap_samples_per_frame = floor(samples_per_frame * overlap_fraction_between_frames);
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
hamming_window = make_column(hamming(samples_per_frame));
normalization_factor = ( hamming_window'* hamming_window)/ samples_per_frame;
FFT_size = samples_per_frame;

%Get initial noise-only data samples and create a windowed matrix out of them:
number_of_initial_seconds_containing_only_noise = 0.12;
number_of_initial_samples_containing_only_noise = round(Fs*number_of_initial_seconds_containing_only_noise);
initial_samples_containing_only_noise = input_signal(1:number_of_initial_samples_containing_only_noise);

%Calculate Sine-Tapers:
number_of_sine_tapers=16;
individual_sine_taper_length = FFT_size;
tapers = zeros(number_of_sine_tapers, individual_sine_taper_length);
for sine_taper_counter = 1:number_of_sine_tapers
    tapers(sine_taper_counter,:) = sqrt(2/(individual_sine_taper_length+1)) * sin(pi*sine_taper_counter*[1:individual_sine_taper_length]'/(individual_sine_taper_length+1));
end 
digamma_function_value = digamma_function(number_of_sine_tapers);

%Calculate the multi-taper noise PSD with sine tapers: 
average_noise_power_spectrum = fft_calculate_sine_multi_tapered_PSD( initial_samples_containing_only_noise, tapers);
   
%Calculate the noise statistics postulated by Walden:
M= 2^nextpow2(samples_per_frame); 
N_autoc = trigamma_function( number_of_sine_tapers)* ( 1 - [0:number_of_sine_tapers+1]/(number_of_sine_tapers+1) );
N_autoc( M/ 2+ 1)= 0;
Sigma_N_firstrow = [N_autoc( 1: M/ 2+ 1), fliplr( N_autoc( 2: M/ 2))];
noise_statistics_postulated_by_walden = real(fft(Sigma_N_firstrow));

%Calculate the log noise power spectrum according to Loizou's paper:
initial_log_noise_power_spectrum = log(average_noise_power_spectrum) - digamma_function_value + log(number_of_sine_tapers);
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialize wavelet-threshold parameters:
%Name of the wavelet used to generate wavelet filters:
wavelet_name='db4';
%Wavelet Threshold type, (scale-dependent='d' , scale-independent='i'):
wavelet_threshold_type='ds';
%Threshold function type: (soft='s' , hard='h'):
wavelet_threshold_function_type='s'; 
%Decomposition Level:
number_of_decomposition_levels=5; 

%Construct decomposition and reconstruction filters based on chosen mother wavelet using Matlab's "wfilters":
[decomposition_filter_low_pass, decomposition_filter_high_pass, reconstruction_filter_low_pass, reconstruction_filter_high_pass] = wfilters(wavelet_name);
wavelet_filters = [decomposition_filter_low_pass; decomposition_filter_high_pass; reconstruction_filter_low_pass; reconstruction_filter_high_pass];
  
%Perform Wavelet Thresholding on the noise average spectrum to remove estimation error assuming it's gaussian:
denoised_log_noise_power_spectrum = perform_wavelet_thresholding( initial_log_noise_power_spectrum, noise_statistics_postulated_by_walden, wavelet_threshold_type, ...
    wavelet_threshold_function_type, wavelet_filters, number_of_decomposition_levels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
%Center log noise power spectrum to make it regular two sided:
denoised_log_noise_power_spectrum = [denoised_log_noise_power_spectrum( 1: samples_per_frame/ 2+ 1); ...
    flipud( denoised_log_noise_power_spectrum( 2: samples_per_frame/ 2))];

%Calculate Linear denoised average noise power spectrum:
average_noise_power_spectrum = exp(denoised_log_noise_power_spectrum);
 
%Get only noisy speech after initial noise-only samples:
input_signal = input_signal(number_of_initial_samples_containing_only_noise+1 : end);
noisy_speech_number_of_samples = length(input_signal);
noisy_speech_number_of_frames = fix( (noisy_speech_number_of_samples-overlap_samples_per_frame) / non_overlapping_samples_per_frame);
initial_indices_of_frames_in_noisy_signal = 1 + (0:(noisy_speech_number_of_frames-1))*non_overlapping_samples_per_frame;
number_of_samples_resulting_from_buffering_input_signal = samples_per_frame + initial_indices_of_frames_in_noisy_signal(noisy_speech_number_of_frames) - 1;
if noisy_speech_number_of_samples < number_of_samples_resulting_from_buffering_input_signal
   %Zero pad signal to equate number of samples of buffered and original signal if necessary:
   noisy_speech(noisy_speech_number_of_samples+1 : samples_per_frame+initial_indices_of_frames_in_noisy_signal(noisy_speech_number_of_frames)-1) = 0;  
end

%Initialize helper variable which contains the overlapping samples of the previous frame:
enhanced_speech_overlapped_samples_from_previous_frame = zeros(overlap_samples_per_frame,1);

%Initialize final enhanced speech with zeros:
final_enhanced_speech = zeros(noisy_speech_number_of_frames * non_overlapping_samples_per_frame,1);

%Decide on over substraction factor:
minimum_SNR_dB = -5;
maximum_SNR_dB = 20;
over_substraction_factor_for_minimum_SNR = 1;
over_substraction_factor_for_maximum_SNR = 5;
over_substraction_factor_constant_SNR_slope = (over_substraction_factor_for_maximum_SNR - over_substraction_factor_for_minimum_SNR )/25;
over_substraction_factor_constant = over_substraction_factor_for_minimum_SNR + 20*over_substraction_factor_constant_SNR_slope;

%Initial algorithm variables:
smoothing_factor_in_noise_spectrum_update = 0.98;
smoothing_factor_in_apriori_SNR_update = 0.98;
VAD_threshold = 0.15;  
spectral_substraction_floor = 0.002;


%Loop over noisy speech frames:
for frame_counter = 1:noisy_speech_number_of_frames
    tic
    %Get start and stop indices of current frame:
    start_index = initial_indices_of_frames_in_noisy_signal(frame_counter);
    stop_index = initial_indices_of_frames_in_noisy_signal(frame_counter) + samples_per_frame - 1;
    
    %Get noisy speech frame:
    current_frame = input_signal(start_index : stop_index);
    
    %Get current frame fft, magnitude, phase and spectrum:
    current_frame_fft = fft(current_frame.* hamming_window,FFT_size);  
    current_frame_PSD = abs(current_frame_fft).^2; 
    current_frame_PSD_window_normalized = current_frame_PSD/(norm(hamming_window)^2);
    
    %Estimat  the noisy speech power spectrum using sine multi tapering:
    current_frame_multi_tapered_PSD = fft_calculate_sine_multi_tapered_PSD( current_frame, tapers );
    
    %Calculate the log noise power spectrum according to Loizou's paper:
    current_frame_log_power_spectrum = log(current_frame_multi_tapered_PSD) - digamma_function_value + log(number_of_sine_tapers);
    
    %Perform Wavelet Thresholding on the noise average spectrum to remove estimation error assuming it's gaussian:
    current_frame_log_power_spectrum_denoised = perform_wavelet_thresholding( current_frame_log_power_spectrum, noise_statistics_postulated_by_walden, wavelet_threshold_type, ...
        wavelet_threshold_function_type, wavelet_filters, number_of_decomposition_levels);
    
    %Center log noise power spectrum to make it regular two sided:
    current_frame_log_power_spectrum_denoised = [current_frame_log_power_spectrum_denoised( 1: samples_per_frame/ 2+ 1); ...
        flipud( current_frame_log_power_spectrum_denoised( 2: samples_per_frame/ 2))];
    
    %Calculate Linear denoised average noise power spectrum:
    currrent_frame_linear_average_power_spectrum_denoised = exp(current_frame_log_power_spectrum_denoised);
    
    %aposteriori SNR estimate:
    aposteriori_SNR_estimate_per_frequency_current = current_frame_PSD_window_normalized ./ average_noise_power_spectrum;
    aposteriori_prime_current = max(aposteriori_SNR_estimate_per_frequency_current - 1 , 0);
    
    %apriori SNR estimate: 
    if frame_counter==1
       apriori_SNR_estimate_per_frequency_current_using_gain_function = 1; 
    else
       apriori_SNR_estimate_per_frequency_current_using_gain_function = (gain_function_previous.^2) .* aposteriori_SNR_estimate_per_frequency_previous;  
    end
     
    %Smooth apriori SNR estimate and calculate respective aposteriori smoothed SNR estimate:
    apriori_SNR_estimate_smoothed = smoothing_factor_in_apriori_SNR_update * apriori_SNR_estimate_per_frequency_current_using_gain_function ...
                                  + (1-smoothing_factor_in_apriori_SNR_update) * aposteriori_prime_current;
    aposteriori_SNR_estimate_smoothed = apriori_SNR_estimate_smoothed + 1;
    
    %VAD decide whether speech is present and if not then update noise spectrum:
    log_likelihood_per_frequency = (aposteriori_SNR_estimate_per_frequency_current./aposteriori_SNR_estimate_smoothed).*apriori_SNR_estimate_smoothed - log(aposteriori_SNR_estimate_smoothed);
    vad_decision(frame_counter) = sum(log_likelihood_per_frequency) / samples_per_frame;   
    if (vad_decision(frame_counter) < VAD_threshold) 
        average_noise_power_spectrum = smoothing_factor_in_noise_spectrum_update*average_noise_power_spectrum + (1- smoothing_factor_in_noise_spectrum_update)*currrent_frame_linear_average_power_spectrum_denoised;
        vad_over_time( start_index : stop_index ) = 0;
    else
        vad_over_time( start_index : stop_index ) = 1;
    end 
    
    %Estimate current Linear, denoised, spectrally substracted, clean speech power power spectrum:
    current_clean_speech_power_spectrum_estimate = currrent_frame_linear_average_power_spectrum_denoised - average_noise_power_spectrum;
    
    %Floor cleaned speech power spectrum to minimum spectral floor: 
    current_clean_speech_power_spectrum_estimate = max( current_clean_speech_power_spectrum_estimate , spectral_substraction_floor*currrent_frame_linear_average_power_spectrum_denoised);

	%Compute the masking threshold: 
	noise_masking_threshold = audio_get_masking_threshold( current_clean_speech_power_spectrum_estimate(1:FFT_size/2+1), FFT_size, Fs, 16);
	noise_masking_threshold = [noise_masking_threshold ; flipud( noise_masking_threshold( 2: FFT_size/ 2))];
	perceptual_noise_masking_ratio_T = average_noise_power_spectrum ./ noise_masking_threshold;
    
	%Calculate gain function:
	gain_function_current = 1 ./ (1 + max( sqrt(perceptual_noise_masking_ratio_T)-1 , 0 ) );
    
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
wavwrite( final_enhanced_speech, Fs, 16, output_file_name);
 
    



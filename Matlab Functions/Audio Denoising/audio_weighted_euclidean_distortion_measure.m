function audio_weighted_euclidean_distortion_measure(input_file_name,output_file_name,cost_function_p_power)

%this is another generalization of the Bayesian estimation process.
%a way to generalize, for instance, the MMSE estimator is to define the:
%(1). Bayesian Cost Function = (Xk-Xk')*Xk^p
%(2). Bayesian Risk: integral[ (Bayesian Cost Function) * P(Xk|Y(wk)) * dXk ]

%(*) to get regular MMSE we need p=0


%Input variables:
input_file_name = 'rodena blabla.wav';
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio170-3500[Hz]lsq smoother with deframing';
output_file_name = 'rodena_speaking_spectral_subtraction.wav';
cost_function_p_power = -1;

%Read input file:
[input_signal, Fs, bits] = wavread(input_file_name);
input_signal = input_signal(:,1);
input_signal = make_column(input_signal);
input_signal = add_noise_of_certain_SNR(input_signal,3,1,0);

%Audio parameters:
frame_size_in_seconds = 0.02;   
samples_per_frame = make_even(floor(Fs*frame_size_in_seconds),1);
overlap_samples_per_frame = samples_per_frame/ 2;	
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
hanning_window = make_column(hanning(samples_per_frame));
hanning_window = hanning_window * overlap_samples_per_frame / sum(hanning_window);
normalization_factor = ( hanning_window'* hanning_window)/ samples_per_frame;
FFT_size = 2*samples_per_frame; 

%Get initial noise-only data samples and create a windowed matrix out of them:
number_of_initial_seconds_containing_only_noise = 0.12;
number_of_initial_samples_containing_only_noise = round(Fs*number_of_initial_seconds_containing_only_noise);
initial_samples_containing_only_noise = input_signal(1:number_of_initial_samples_containing_only_noise);
initial_noise_data_matrix = buffer(initial_samples_containing_only_noise,samples_per_frame,0);
initial_noise_data_matrix = bsxfun(@times,initial_noise_data_matrix,hanning_window);
average_noise_power_spectrum = mean( abs(fft(initial_noise_data_matrix,FFT_size)) , 2);
average_noise_power_spectrum = average_noise_power_spectrum.^2;
 
%Get only noisy speech after initial noise-only samples: 
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

%Initialize final enhanced speech:
total_number_of_frames = floor(length(input_signal)/non_overlapping_samples_per_frame)-1;
final_enhanced_speech = zeros(total_number_of_frames*non_overlapping_samples_per_frame,1);

%Set parameter values:
smoothing_factor_in_noise_spectrum_update = 0.98;
smoothing_factor_in_apriori_SNR_update = 0.98;
VAD_threshold = 0.15; 
apriori_SNR_estimate_minimum = 10^(-25/10);


%Loop over noisy frames and enhance them:
for frame_counter = 1:total_number_of_frames
    tic
    %Get start and stop indices of current frame:
    start_index = initial_indices_of_frames_in_noisy_signal(frame_counter);
    stop_index = initial_indices_of_frames_in_noisy_signal(frame_counter) + samples_per_frame - 1;
    
    %Get noisy speech frame:
    current_frame = noisy_speech(start_index : stop_index);
    
    %Window current frame:
    current_frame = current_frame .* hanning_window;
    
    %Get current frame fft, magnitude, phase and spectrum:
    current_frame_fft = fft(current_frame,FFT_size);
    current_frame_fft_magnitude = abs(current_frame_fft);
    current_frame_fft_phase  = angle(current_frame_fft);
    current_frame_PSD = current_frame_fft_magnitude.^2;
     
    %aposteriori SNR estimate:
    aposteriori_SNR_estimate_per_frequency_current = min(current_frame_PSD ./ average_noise_power_spectrum,40);
    aposteriori_prime_current = max(aposteriori_SNR_estimate_per_frequency_current-1,0); 
    
    %apriori SNR estimate:
    if frame_counter==1
       apriori_SNR_estimate_per_frequency_using_previous_gain_function = 1; 
    else
       apriori_SNR_estimate_per_frequency_using_previous_gain_function = (gain_function_previous.^2) .* aposteriori_SNR_estimate_per_frequency_previous;  
    end
    
    %Smooth apriori SNR estimate and calculate respective aposteriori smoothed SNR estimate:
    apriori_SNR_estimate_smoothed = smoothing_factor_in_apriori_SNR_update * apriori_SNR_estimate_per_frequency_using_previous_gain_function ...
                                  + (1-smoothing_factor_in_apriori_SNR_update) * aposteriori_prime_current;
    apriori_SNR_estimate_smoothed = max(apriori_SNR_estimate_minimum,apriori_SNR_estimate_smoothed); 
    aposteriori_SNR_estimate_smoothed = apriori_SNR_estimate_smoothed + 1;
     
    %VAD decide whether speech is present and if not then update noise spectrum:
    log_likelihood_per_frequency = (aposteriori_SNR_estimate_per_frequency_current./aposteriori_SNR_estimate_smoothed).*apriori_SNR_estimate_smoothed - log(aposteriori_SNR_estimate_smoothed);
    vad_decision(frame_counter) = sum(log_likelihood_per_frequency) / samples_per_frame;   
    if (vad_decision(frame_counter) < VAD_threshold) 
        average_noise_power_spectrum = smoothing_factor_in_noise_spectrum_update*average_noise_power_spectrum + (1- smoothing_factor_in_noise_spectrum_update)*current_frame_PSD;
        vad_over_time( start_index : stop_index ) = 0;
    else
        vad_over_time( start_index : stop_index ) = 1;
    end
    
    %Assign simplifying variables for the following calculations:
    gamma_k = aposteriori_SNR_estimate_per_frequency_current;
    xi_k = apriori_SNR_estimate_smoothed;
    v_k = (xi_k./(1+xi_k)) .* gamma_k;

    %Calculate gain function:
    if cost_function_p_power == -1
       gain_function_current =  gamma((cost_function_p_power+3)/2)/gamma(cost_function_p_power/2+1) .* (sqrt(v_k)./gamma_k) ./ exp(-v_k/2) ./ besseli(0,v_k/2);
    else
       gain_function_numerator = gamma((cost_function_p_power+3)/2)/gamma(cost_function_p_power/2+1) .* sqrt(v_k) .* confhyperg(-(cost_function_p_power+1)/2,1,-v_k,100);
       gain_function_denominator = gamma_k .* confhyperg(-cost_function_p_power/2,1,-v_k,100);
       gain_function_current = gain_function_numerator ./ gain_function_denominator;
    end
    
    %Calculate Enhanced speech frame: 
    enhanced_frame_current = real(ifft( current_frame_fft_magnitude .* gain_function_current .* exp(1i*current_frame_fft_phase), FFT_size));

    %Overlap-Add:    
    first_overlapping_part_of_enhanced_signal = enhanced_frame_current(1:overlap_samples_per_frame) + enhanced_speech_overlapped_samples_from_previous_frame;
    second_nonoverlapping_part_of_enhanced_signal = enhanced_frame_current(overlap_samples_per_frame+1 : samples_per_frame);
    current_final_enhanced_speech_frame = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
    final_enhanced_speech(start_index:stop_index) = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
     
    %Remember overlapping part for next overlap-add:
    enhanced_speech_overlapped_samples_from_previous_frame = enhanced_frame_current( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame);
    
    %Remember current wiener filter gain function for later apriori SNR estimation: 
    gain_function_previous = gain_function_current; 
    aposteriori_SNR_estimate_per_frequency_previous = aposteriori_SNR_estimate_per_frequency_current;
    toc 
end
     
%Write down final enhanced speech:
sound(final_enhanced_speech,Fs);
wavwrite(final_enhanced_speech,Fs,16,output_file_name);


% function audio_spectral_substraction_with_adapative_averaging( input_file_name,output_file_name)

%Input variables:
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio170-3500[Hz]lsq smoother with deframing';
output_file_name = 'rodena_speaking_spectral_subtraction.wav';

%Read input file:
[input_signal, Fs, nbits] = wavread( input_file_name);
input_signal = make_row(input_signal);

%Decide on frame and sub-frame parameters:
if Fs == 8000
    frame_size=160; sub_frame_size=32; FFT_size=256;
elseif Fs == 16000
    frame_size=320; sub_frame_size=64; FFT_size=512;
elseif Fs == 44100
    frame_size=320*3; sub_frame_size=64*3; FFT_size=512*3;
else
    exit( 'incorrect sampling rate!\n');
end

%Initialize frame, sub-frame, and window parameters:
total_number_of_frames = floor(length(input_signal)/frame_size);
number_of_sub_frames_in_one_frame = frame_size / sub_frame_size; % number of segments of M
hanning_window = make_row(hanning(frame_size)); % hanning window
final_enhanced_speech = zeros(1,(total_number_of_frames-1)*frame_size + FFT_size);

%Smoothing parameters values:
smoothing_factor_in_noise_spectrum_update = 0.98; % smoothing factor in noise spectrum update
smoothing_factor_in_apriori_update = 0.98; % smoothing factor in priori update

%VAD threshold:
VAD_threshold = 0.15; % VAD threshold

%Gain function parameters:
gain_function_smoother_smoothing_parameter = 0.8; % smoothing factor in G_M update
gain_function_noise_oversubstraction_factor = 0.7; % oversubtraction factor

%Get initial frames containing only noise:
number_of_initial_seconds_containing_only_noise = 0.12;
number_of_initial_samples_containing_only_noise = Fs * number_of_initial_seconds_containing_only_noise;
initial_samples_containing_only_noise = input_signal( 1: number_of_initial_samples_containing_only_noise);
number_of_sub_frames_containing_only_noise = number_of_initial_samples_containing_only_noise / sub_frame_size;  % L is 20ms

%Estimate Noise power spectrum using the Barlett Method:
average_noise_power_spectrum = zeros( 1, sub_frame_size);
for frame_counter= 1: number_of_sub_frames_containing_only_noise
    current_noise_sub_frame = initial_samples_containing_only_noise( (frame_counter-1)*sub_frame_size+1 : frame_counter*sub_frame_size);
    current_noise_fft = fft(current_noise_sub_frame , sub_frame_size);
    average_noise_power_spectrum = average_noise_power_spectrum + abs(current_noise_fft).^ 2/sub_frame_size;
end
average_noise_power_spectrum = average_noise_power_spectrum / number_of_sub_frames_containing_only_noise;
average_noise_fft_magnitude = sqrt(average_noise_power_spectrum);


%Go over Noisy speech and Enhance it:
flag_use_window = 1;
for frame_counter = 1:total_number_of_frames
    tic
    %Get current frame:
    current_noisy_frame = input_signal( (frame_counter-1)*frame_size+1 : frame_counter*frame_size);    
    
    %Use window if wanted:
    if flag_use_window==1; current_noisy_frame = current_noisy_frame.*hanning_window; end
    
    %Estimate current noisy speech power spectrum using the Barlett Method:
    current_average_noisy_speech_power_spectrum = zeros(1,sub_frame_size);
    for n = 1:number_of_sub_frames_in_one_frame
        current_noisy_speech_sub_frame= current_noisy_frame( (n-1)* sub_frame_size+1 : n*sub_frame_size );
        current_noisy_speech_sub_frame_fft = fft( current_noisy_speech_sub_frame, sub_frame_size);
        current_average_noisy_speech_power_spectrum = current_average_noisy_speech_power_spectrum + abs(current_noisy_speech_sub_frame_fft).^2/sub_frame_size;
    end
    current_average_noisy_speech_power_spectrum = current_average_noisy_speech_power_spectrum / number_of_sub_frames_in_one_frame; 
    current_average_noisy_speech_fft_magnitude = sqrt(current_average_noisy_speech_power_spectrum); 
    
    %VAD - make a decision as to whether speech is present and whether to update the noise spectrum:
    if (frame_counter==1) % initialize posteri
        aposteriori_SNR_current = (current_average_noisy_speech_fft_magnitude.^2) ./ (average_noise_fft_magnitude.^2);
        aposteriori_prime = max(aposteriori_SNR_current - 1,0); 
        apriori_SNR_current_smoothed_estimate = smoothing_factor_in_apriori_update + (1-smoothing_factor_in_apriori_update)*aposteriori_prime;
        aposteriori_SNR_current_smoothed_estimate = apriori_SNR_current_smoothed_estimate + 1;
    else
        aposteriori_SNR_old = aposteriori_SNR_current;
        aposteriori_SNR_current = (current_average_noisy_speech_fft_magnitude.^2) ./ (average_noise_fft_magnitude.^2);
        aposteriori_prime = max(aposteriori_SNR_current-1,0);
        apriori_SNR_estimate_old = (smoothed_gain_function_G.^2).*aposteriori_SNR_old;
        apriori_SNR_current_smoothed_estimate = smoothing_factor_in_apriori_update*apriori_SNR_estimate_old + (1-smoothing_factor_in_apriori_update)*aposteriori_prime;      
        aposteriori_SNR_current_smoothed_estimate = apriori_SNR_current_smoothed_estimate + 1;
    end
    %Use the likelihood-ratio approach to make a decision as to whether there's speech present:
    log_likelihood_per_bin = (aposteriori_SNR_current./aposteriori_SNR_current_smoothed_estimate).*apriori_SNR_current_smoothed_estimate - log(aposteriori_SNR_current_smoothed_estimate);
    vad_decision(frame_counter) = sum(log_likelihood_per_bin) / sub_frame_size;    
    
    %Update Noise spectrum if VAD decides there's no speech:
    if (vad_decision(frame_counter)< VAD_threshold)
        average_noise_fft_magnitude = smoothing_factor_in_noise_spectrum_update*average_noise_fft_magnitude + (1-smoothing_factor_in_noise_spectrum_update)*current_average_noisy_speech_fft_magnitude;
        vad_decision_over_time( (frame_counter-1)*frame_size+1 : frame_counter*frame_size )= 0;
    else
        vad_decision_over_time( (frame_counter-1)*frame_size+1 : frame_counter*frame_size )= 1;
    end
    
    %Update and smooth noisy speech Gain function:
    current_smoothed_noisy_to_enhanced_speech_gain_function = max(1 - gain_function_noise_oversubstraction_factor*(average_noise_fft_magnitude./current_average_noisy_speech_fft_magnitude),0);
    
    %Calculate spectral discrepancy measure (proxy of stationarity) beta as a prelude to adaptive gain caclulation:
    beta_spectral_discrepancy = min(1, sum(abs(current_average_noisy_speech_fft_magnitude-average_noise_fft_magnitude))/sum(average_noise_fft_magnitude));
    alpha_1 = 1 - beta_spectral_discrepancy;    
     
    %Initialize gain function G smoother constant:
    if (frame_counter== 1)
        adaptive_gain_function_smoother = alpha_1; 
    end
    
    %Update adaptive (smoothed) gain function smoother coefficient:
    if (adaptive_gain_function_smoother < alpha_1)
        adaptive_gain_function_smoother = (1-gain_function_smoother_smoothing_parameter)*alpha_1 + gain_function_smoother_smoothing_parameter*adaptive_gain_function_smoother;
    else
        adaptive_gain_function_smoother = alpha_1;
    end
    
    %Calculate smoothed gain function G with the calculated adaptive gain smoother coefficient:
    if (frame_counter == 1)
        smoothed_gain_function_G = (1-adaptive_gain_function_smoother)*current_smoothed_noisy_to_enhanced_speech_gain_function;
    else
        smoothed_gain_function_G = adaptive_gain_function_smoother*smoothed_gain_function_G + (1-adaptive_gain_function_smoother)*current_smoothed_noisy_to_enhanced_speech_gain_function;
    end
    
    %Interpolate smoothed gain function over from sub-frame samples to frame samples: 
    smoothed_gain_function_G_impulse_response = firls(sub_frame_size, (0:sub_frame_size-1)/sub_frame_size, smoothed_gain_function_G);    
    interpolated_smoothed_gain_function = fft( smoothed_gain_function_G_impulse_response, FFT_size);    
    
    %Use smoothed gain function to Filter noisy speech frame to make the Enhanced speech frame:    
    enhanced_speech_current_frame = ifft( fft(current_noisy_frame,FFT_size) .* interpolated_smoothed_gain_function, FFT_size);

    %Overlap-ADD:
    if (frame_counter == 1)
        final_enhanced_speech( 1: FFT_size)= enhanced_speech_current_frame;
    else
        overlap= enhanced_speech_current_frame( 1: FFT_size- frame_size)+ final_enhanced_speech( (frame_counter-1)* ...
            frame_size+ 1: (frame_counter-1)*frame_size+ FFT_size- frame_size);
        final_enhanced_speech( (frame_counter-1)*frame_size+ 1: (frame_counter-1)*frame_size+ FFT_size- frame_size)= overlap;
        final_enhanced_speech( (frame_counter-1)*frame_size+ FFT_size- frame_size+ 1: (frame_counter-1)*frame_size+ FFT_size)= ...
            enhanced_speech_current_frame( FFT_size- frame_size+ 1: FFT_size);
    end    
    toc
end %end of frames loop
 
%Write Enhanced speech to .wav file:
sound(final_enhanced_speech,Fs); 
wavwrite( final_enhanced_speech, Fs, nbits, output_file_name);


    

% function audio_wiener_filter_sub_space( input_file_name, output_file_name)

%Input variables:
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio170-3500[Hz]lsq smoother with deframing';
output_file_name = 'rodena_speaking_spectral_subtraction.wav';

%Read input file:
[input_signal, Fs, bits] = wavread(input_file_name);
input_signal = make_column(input_signal);
% input_signal = add_noise_of_certain_SNR(input_signal,-20,1,0);

%Sub-frame parameters:
sub_frame_duration_in_seconds = 0.004;
samples_per_sub_frame = make_even(floor(Fs*sub_frame_duration_in_seconds),1); 
subframe_window = hamming(samples_per_sub_frame); 

%Frame parameters:
frame_size_in_seconds = 0.032;   
overlap_fraction_between_frames = 0.5;
samples_per_frame = round(frame_size_in_seconds/sub_frame_duration_in_seconds) * samples_per_sub_frame;
overlap_samples_per_frame = floor(samples_per_frame * overlap_fraction_between_frames);
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
frame_window = make_column(hamming(samples_per_frame));
normalization_factor = ( frame_window'* frame_window)/ samples_per_frame;
FFT_size = samples_per_frame;
number_of_subframes_in_frame = samples_per_frame/samples_per_sub_frame;

%Initialize helper variable which contains the overlapping samples of the previous frame:
enhanced_speech_overlapped_samples_from_previous_frame = zeros(overlap_samples_per_frame,1);

%Get initial noise-only data samples and create a windowed matrix out of them:
number_of_initial_seconds_containing_only_noise = 0.5;
number_of_initial_samples_containing_only_noise = round(Fs*number_of_initial_seconds_containing_only_noise);
initial_samples_containing_only_noise = input_signal(1:number_of_initial_samples_containing_only_noise);

%Decide on over substraction factor:
minimum_SNR_dB = -5;
maximum_SNR_dB = 20;
over_substraction_factor_for_minimum_SNR = 1;
over_substraction_factor_for_maximum_SNR = 5;
over_substraction_factor_constant_SNR_slope = (over_substraction_factor_for_maximum_SNR - over_substraction_factor_for_minimum_SNR )/25;
over_substraction_factor_constant = over_substraction_factor_for_minimum_SNR + 20*over_substraction_factor_constant_SNR_slope;

%Initial algorithm variables:
number_of_sine_tapers = 16; 
VAD_threshold = 1.2; 
smoothing_factor_in_noise_covariance_matrix_update = 0.98; 
In = eye(samples_per_sub_frame);
signal_sine_tapers = fft_get_sine_tapers( number_of_sine_tapers, samples_per_frame );

%Get noise toeplitz form covariance matrix:
if number_of_sine_tapers== 1
    %Get noise samples auto-correlation function:
    noise_autocorrelation = xcorr(initial_samples_containing_only_noise, samples_per_sub_frame - 1, 'biased');
    
    %Form a Toeplitz matrix to get noise covariance matrix:
    Rn = toeplitz( noise_autocorrelation( samples_per_sub_frame: end));
else
    %Get multi tapered noise covariance matrix:
    individual_sine_taper_length = number_of_initial_samples_containing_only_noise;
    noise_sine_tapers = fft_get_sine_tapers(number_of_sine_tapers,individual_sine_taper_length);
    Rn = fft_get_multi_tapered_covariance_matrix( initial_samples_containing_only_noise, samples_per_sub_frame, noise_sine_tapers);
end
iRn = inv(Rn);
 
%Get only noisy speech after initial noise-only samples:
noisy_speech_number_of_samples = length(input_signal);
noisy_speech_number_of_frames = fix( (noisy_speech_number_of_samples-overlap_samples_per_frame) / non_overlapping_samples_per_frame);
initial_indices_of_frames_in_noisy_signal = 1 + (0:(noisy_speech_number_of_frames-1))*non_overlapping_samples_per_frame;
number_of_samples_resulting_from_buffering_input_signal = samples_per_frame + initial_indices_of_frames_in_noisy_signal(noisy_speech_number_of_frames) - 1;
if noisy_speech_number_of_samples < number_of_samples_resulting_from_buffering_input_signal
   %Zero pad signal to equate number of samples of buffered and original signal if necessary:
   input_signal(noisy_speech_number_of_samples+1 : samples_per_frame+initial_indices_of_frames_in_noisy_signal(noisy_speech_number_of_frames)-1) = 0;  
end
 



%Loop over noisy speech frames:
for frame_counter = 1:noisy_speech_number_of_frames  
    tic
    %Get start and stop indices of current frame:
    start_index = initial_indices_of_frames_in_noisy_signal(frame_counter);
    stop_index = initial_indices_of_frames_in_noisy_signal(frame_counter) + samples_per_frame - 1;
    
    %Get noisy speech frame:
    current_frame = input_signal( start_index:stop_index );    
    
    %Use sine tapers to estimate current frame covariance matrix:
    if number_of_sine_tapers == 1
        current_frame_auto_correlation = xcorr( current_frame, samples_per_sub_frame - 1, 'biased' );
        Ry = toeplitz( current_frame_auto_correlation( samples_per_sub_frame : 2*samples_per_sub_frame - 1));
    else
        Ry = fft_get_multi_tapered_covariance_matrix( current_frame, samples_per_sub_frame, signal_sine_tapers);
    end  
       
    %Use simple VAD to update the noise cov matrix, Rn, and subsequently, iRn: 
    vad_ratio = Ry(1,1)/Rn(1,1); 
    if (vad_ratio <= VAD_threshold)
        Rn = smoothing_factor_in_noise_covariance_matrix_update*Rn + (1-smoothing_factor_in_noise_covariance_matrix_update)*Ry;
        iRn = inv(Rn);
    end 
    
    %Estimate (Rn)^-1*Rx = Rn^-1*(Ry-Rn) = Rn^-1*Ry - I  for subsequent EVD which will enable simultaneous
    %diagonalization of both Rx and Rn:
    iRnRx = iRn*Ry - In;
       
    %Eigen Value Decomposition of (Rn^-1)*Rx:
    [V, D] = eig(iRnRx);  
    iV = inv(V); 

    %Rx/Rn eigenvalues sum (besides those below 0 which are considered out
    %of the speech signal subspace) can give an estimate of the SNR:
    iRnRx_speech_subspace_eigenvalues = max(diag(D),0);
    SNR = sum(iRnRx_speech_subspace_eigenvalues) / samples_per_sub_frame;
    current_final_apriori_log_SNR_estimate = 10*log10( SNR + eps);
     
    %Decide on current lagrange multiplier used in the gain function:
    if current_final_apriori_log_SNR_estimate >= 20
        lagrange_factor_current = over_substraction_factor_for_minimum_SNR;
    elseif current_final_apriori_log_SNR_estimate < -5
        lagrange_factor_current = over_substraction_factor_for_maximum_SNR;
    else 
        lagrange_factor_current = over_substraction_factor_constant - current_final_apriori_log_SNR_estimate*over_substraction_factor_constant_SNR_slope;
    end
    
    %Calculate estimator values after transformation by V (prewhittening):
    gain_whittened_matrix_values = iRnRx_speech_subspace_eigenvalues ./ (iRnRx_speech_subspace_eigenvalues + lagrange_factor_current);   
    G = diag(gain_whittened_matrix_values);
    
    %Calculate gain function matrix (prewhitten -> use optimal G estimator -> unwhitten):
    H_gain_function_current = iV'*G*V'; 
    
    %Synthesis:
    %(1). Sub-frames / Inter-frame fusion:
    %Initial sub-frame indices:
    sub_frame_start = 1;
    sub_frame_stop = samples_per_sub_frame;
    %Initialize subframe-overlap frame for overlap-add:
    sub_frame_overlap = zeros( samples_per_sub_frame/2, 1);
    for subframe_counter = 1:(2*round(number_of_subframes_in_frame)-1)
        %get current sub-frame:
        current_sub_frame = current_frame( sub_frame_start : sub_frame_stop );
        %enhance current sub-frame:
        enhanced_sub_frame_current = (H_gain_function_current*current_sub_frame) .* subframe_window;
        %overlapp add: 
        enhanced_sub_frames_fusion( sub_frame_start : sub_frame_start+samples_per_sub_frame/2-1 )= ...
            enhanced_sub_frame_current( 1 : samples_per_sub_frame/2 ) + sub_frame_overlap; 
        sub_frame_overlap = enhanced_sub_frame_current( samples_per_sub_frame/2+1 : samples_per_sub_frame);
        %update sub-frame indices:
        sub_frame_start = sub_frame_start + samples_per_sub_frame/2;
        sub_frame_stop = sub_frame_stop + samples_per_sub_frame/2;
    end
    enhanced_sub_frames_fusion( sub_frame_start : sub_frame_start+samples_per_sub_frame/2-1) = sub_frame_overlap; 
     
    %(2). Large frame / Across-frame fusion:
    enhanced_frame_current = enhanced_sub_frames_fusion' .* frame_window;
    first_overlapping_part_of_enhanced_signal = enhanced_frame_current(1:overlap_samples_per_frame) + enhanced_speech_overlapped_samples_from_previous_frame;
    second_nonoverlapping_part_of_enhanced_signal = enhanced_frame_current(overlap_samples_per_frame+1:samples_per_frame);
    final_enhanced_speech(start_index:stop_index) = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];  
    
    %Remember overlapping part for next overlap-add:
    enhanced_speech_overlapped_samples_from_previous_frame = enhanced_frame_current( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame);
    
    toc
end
final_enhanced_speech( start_index: start_index + overlap_samples_per_frame - 1) = enhanced_speech_overlapped_samples_from_previous_frame; 

%Write final enhanced speech to .wav file:
sound(final_enhanced_speech,Fs); 
wavwrite(final_enhanced_speech, Fs, Nbits, output_file_name);
 

 
 





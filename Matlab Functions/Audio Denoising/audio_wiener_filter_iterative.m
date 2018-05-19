% function [enhanced_signal] = audio_wiener_filter_iterative(input_file_name,output_file_name, number_of_wiener_filter_iterations)

%Input variables:
input_file_name = '900m, reference loudness=69dBspl, conversation, laser power=1 , Fs=4935Hz, channel2.wav';
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio150-3000[Hz]';
output_file_name = 'rodena_speaking_spectral_subtraction.wav';
number_of_wiener_filter_iterations = 2;

%Read input file:
[input_signal, Fs, bits] = wavread( input_file_name);
input_signal = make_column(input_signal);
% input_signal = add_noise_of_certain_SNR(input_signal,10,1,0);

%Audio parameters:
frame_size_in_seconds = 0.02;   
samples_per_frame = make_even(floor(Fs*frame_size_in_seconds),1);

%Get initial frames containing only noise:
number_of_initial_seconds_containing_only_noise = 0.12;
number_of_initial_frames_containing_only_noise = round(Fs * number_of_initial_seconds_containing_only_noise / samples_per_frame);
number_of_initial_samples_containing_only_noise = number_of_initial_frames_containing_only_noise * samples_per_frame;
overlap_samples_per_frame = samples_per_frame/ 2;	
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
hanning_window = make_column(hanning(samples_per_frame));

%Get LPC model order and FFT size:
LPC_model_order = 12;
FFT_size = 2*samples_per_frame;

%Initialize fft exponential factor matrix for later easy a(k)*exp(-2*pi*i*f*k) summation:
for m = 1:LPC_model_order+1
    fft_exp_matrix(m,:) = exp(-2*pi*1i*(m-1)*[0:FFT_size-1]/FFT_size);
end 
 
%Get initial noise-only data samples and create a windowed matrix out of them:
initial_samples_containing_only_noise = input_signal(1:number_of_initial_samples_containing_only_noise);
initial_noise_data_matrix = buffer(initial_samples_containing_only_noise,samples_per_frame,overlap_samples_per_frame);
initial_noise_data_matrix = bsxfun(@times,initial_noise_data_matrix,hanning_window);
average_noise_power_spectrum = mean( abs(fft(initial_noise_data_matrix,FFT_size)).^2 , 2);

%Get only noisy speech after initial noise-only samples:
noisy_speech = input_signal(samples_per_frame*number_of_initial_frames_containing_only_noise+1 : end);	%x is to-be-enhanced noisy speech
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

%Loop over frames and Enhance speech:
for frame_counter = 1:noisy_speech_number_of_frames
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
   current_frame_spectrum = current_frame_fft_magnitude.^ 2;	%power spectrum of noisy speech
   
   %Initialize lpc coefficients for use in the iterative wiener filter from the noisy speech:
   lpc_coeffs = (lpc(current_frame, LPC_model_order))';
   
    
   %Initialize first fft for iterative wiener filtering:
   former_enhanced_speech_fft_estimate = current_frame_fft;
   for iteration_counter = 1:number_of_wiener_filter_iterations
      
      %Estimate original speech power spectrum denominator inside lpc framework: 
      Pxx_lpc_denominator = 1./(abs(fft_exp_matrix'*lpc_coeffs).^ 2);
      Pxx_lpc_mean_energy_without_excitation_gain_g = mean(1./Pxx_lpc_denominator);
      
      %Estimate Pxx estimate from spectral substraction:
      Pxx_spectral_substraction = current_frame_spectrum - average_noise_power_spectrum;
      Pxx_spectral_substraction_mean_energy = mean(Pxx_spectral_substraction);
      
      %Calculate excitation gain g by equating the two Pxx estimates:
      min_g = 10^-16;
      g = max(Pxx_spectral_substraction_mean_energy / Pxx_lpc_mean_energy_without_excitation_gain_g, min_g);
      
      %Estimate original speech power spectrum inside lpc framework:
      Pxx_lpc_including_g = g./Pxx_lpc_denominator;
      
      %Calculate the current iteration Wiener filter:
      current_iteration_wiener_filter = Pxx_lpc_including_g ./ (Pxx_lpc_including_g + average_noise_power_spectrum);
      
      %Calculate enhanced speech signal and fft:
      enhanced_frame_current_fft = former_enhanced_speech_fft_estimate.*current_iteration_wiener_filter;
      enhanced_frame_current = real(ifft(enhanced_frame_current_fft,FFT_size));   
      
      %Update former enhanced speech fft estimate for next iteration:
      former_enhanced_speech_fft_estimate = fft(enhanced_frame_current, FFT_size);
      
      %Recalculate lpc coefficients with for current iterations:
      if iteration_counter ~= number_of_wiener_filter_iterations
         lpc_coeffs = lpc( enhanced_frame_current, LPC_model_order)';   
      end   
   end %end of iterations loop
   
   
   %Overlap-Add:
   first_overlapping_part_of_enhanced_signal = enhanced_frame_current(1:overlap_samples_per_frame) + enhanced_speech_overlapped_samples_from_previous_frame;
   second_nonoverlapping_part_of_enhanced_signal = enhanced_frame_current(overlap_samples_per_frame+1 : samples_per_frame);
   final_enhanced_speech(start_index:stop_index) = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
   
   %Remember overlapping part for next overlap-add:
   enhanced_speech_overlapped_samples_from_previous_frame = enhanced_frame_current( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame);
   toc
end 
   
%Write down final enhanced speech:
sound(final_enhanced_speech,Fs);
wavwrite(final_enhanced_speech, Fs, bits, output_file_name);


 



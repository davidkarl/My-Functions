function [parameters] = noise_estimation_SNR_dependent(current_frame_ps,parameters)

%Initialize parameters:
final_noise_ps_smoothed = parameters.noise_ps;
last_values_of_final_noise_power_spectrums = parameters.last_values_of_final_noise_power_spectrums;
exponential_smoothing_beta = parameters.exponential_smoothing_beta;
frame_refresh_counter = parameters.frame_refresh_counter;
number_of_frames_to_remember = parameters.number_of_frames_to_remember;

aposteriori_SNR = current_frame_ps ./ mean(last_values_of_final_noise_power_spectrums,1)';
final_noise_ps_smoothing_factor = 1./ (1 + exp(-exponential_smoothing_beta*(aposteriori_SNR-1.5)) );
final_noise_ps_smoothed = final_noise_ps_smoothing_factor.*final_noise_ps_smoothed ...
    + (1-final_noise_ps_smoothing_factor).*current_frame_ps;

last_values_of_final_noise_power_spectrums(frame_refresh_counter,:) = final_noise_ps_smoothed;

frame_refresh_counter = frame_refresh_counter + 1;
if frame_refresh_counter == number_of_frames_to_remember
    frame_refresh_counter = 1;
end 
  

%Update final parameters:
parameters.noise_ps = final_noise_ps_smoothed;
parameters.last_values_of_final_noise_power_spectrum = last_values_of_final_noise_power_spectrums;
parameters.exponential_smoothing_beta = exponential_smoothing_beta;
parameters.frame_refresh_counter = frame_refresh_counter;
parameters.number_of_frames_to_remember = number_of_frames_to_remember;


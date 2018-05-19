function [average_noise_fft_magnitude] = fft_calculate_average_power_spectrum_at_file_start(input_signal,number_of_frames_from_start,samples_per_frame,window_to_use,fft_length)

average_noise_power_spectrum_magnitude=zeros(fft_length,1);
current_frame_start = 1;
current_frame_stop = current_frame_start + samples_per_frame - 1;
input_signal = input_signal(:,1); %signle-channel
for k=1:number_of_frames_from_start
    %update noise power spectrum (NO FFT SHIFT):
    current_frame_power_spectrum_magnitude = abs(fft(input_signal(current_frame_start:current_frame_stop).* window_to_use, fft_length)).^2;
    average_noise_power_spectrum_magnitude = average_noise_power_spectrum_magnitude + current_frame_power_spectrum_magnitude;
    
    %update frame locations:
    current_frame_start = current_frame_start + samples_per_frame;
    current_frame_stop = current_frame_stop + samples_per_frame;
end
average_noise_fft_magnitude = sqrt(average_noise_power_spectrum_magnitude/number_of_frames_from_start);



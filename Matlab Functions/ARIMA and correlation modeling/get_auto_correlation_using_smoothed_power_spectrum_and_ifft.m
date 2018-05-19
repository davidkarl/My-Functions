function [auto_correlation] = ...
            get_auto_correlation_using_smoothed_power_spectrum_and_ifft(...
                    input_signal,number_of_coefficients,samples_per_frame,non_overlapping_samples_per_frame)

% Computes the first n coefficients of the autocorrelation
% function of the signal y
% It uses smoothed estimate of the periodograms of length w
% with shift s
% c(1) is the correlation at lag 0, c(2) at lag 1 etc.
% The correlation is estimated using FFT
% Usage: c=corrss(y,n,w,s)

input_signal_length = length(input_signal);
total_number_of_frames = floor((input_signal_length-samples_per_frame)/non_overlapping_samples_per_frame);
hanning_window = hanning(samples_per_frame);
hanning_window = hanning_window * samples_per_frame / sum(hanning_window);
smoothed_power_spectrum = zeros(1,samples_per_frame);

for i = 0:total_number_of_frames
    start_index = i*non_overlapping_samples_per_frame + 1;
    stop_index = i*non_overlapping_samples_per_frame + samples_per_frame;
    smoothed_power_spectrum = smoothed_power_spectrum + abs(fft(input_signal(start_index:stop_index))).^2;
end

auto_correlation = real(ifft(smoothed_power_spectrum / ((total_number_of_frames+1)*samples_per_frame)));
auto_correlation = auto_correlation(1:number_of_coefficients)';






















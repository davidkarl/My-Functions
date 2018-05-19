function [index] = fft_frequency_to_index(fft_signal_length,Fs,frequencies,flag_one_sided)
%ASSUMES FFT IS DOUBLE-SIDED (regular fft):

FFT_length = fft_signal_length; %double-sides fft length
one_point_in_Hz = (Fs)/(FFT_length);
if flag_one_sided==1
    index = round(frequencies/(Fs/2) * (FFT_length/2));
else
    index =  floor(fft_signal_length/2) + mod(fft_signal_length,2) + round(frequencies/(Fs/2) * (FFT_length/2)); %==round(start_frequency/Fs*FFT_length)
end


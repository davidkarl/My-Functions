function [frequency] = fft_index_to_frequency(fft_signal_length,Fs,index)
%ASSUMES FFT IS DOUBLE-SIDED (regular fft):

FFT_length = fft_signal_length; %double-sides fft length
one_point_in_Hz = (Fs)/(FFT_length);
frequency = (Fs/fft_signal_length)*(-ceil((fft_signal_length-1)/2)+(index-1));



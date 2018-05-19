function [bin_start,bin_center,bin_stop,bin_size] = fft_get_linear_frequency_vecs(number_of_frequency_bands,fft_length)
%maybe change start_frequency to 0?

bin_size(1) = floor((fft_length/2)/(number_of_frequency_bands));
bin_start = 1 + [0:1:number_of_frequency_bands-1]*bin_size(1);
bin_stop = [1:1:number_of_frequency_bands]*bin_size(1);
bin_center = (bin_start+bin_stop)/2;


function [bin_start,bin_center,bin_stop,bin_size]=fft_get_log_frequency_vecs(number_of_frequency_bands,Fs,fft_length)
%maybe change start_frequency to 0?

%bin parameters:
start_frequency=1;
stop_frequency=Fs/2;
log_frequency_range=log10(stop_frequency/start_frequency);
log_interval=log_frequency_range/number_of_frequency_bands;

%create bins:
bin_start = start_frequency.*10.^(log_interval*[0:1:number_of_frequency_bands-1]);
bin_stop = start_frequency.*10.^(log_interval*[1:1:number_of_frequency_bands]);
bin_start = round(bin_start*fft_length/Fs)+1;
bin_stop = round(bin_stop*fft_length/Fs)+1;
bin_center = 0.5*(bin_start+bin_stop);
bin_size = bin_stop-bin_start+1;

  
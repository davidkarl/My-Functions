function [bin_start,bin_center,bin_stop,bin_size] = fft_get_mel_frequency_vecs(number_of_frequency_bands,low,high,fft_length,Fs)
% This function returns the lower, center and upper freqs
% of the filters equally spaced in mel-scale
% Input: N - number of filters
% 	 low - (left-edge) 3dB frequency of the first filter
%	 high - (right-edge) 3dB frequency of the last filter
%
% Copyright (c) 1996-97 by Philipos C. Loizou

%Mel basic parameters:
ac=1100; fc=800;

%Mel parameters:
LOW =ac*log(1+low/fc);
HIGH=ac*log(1+high/fc);
N1=number_of_frequency_bands+1;
e1=exp(1);
fmel(1:N1)=LOW+[1:N1]*(HIGH-LOW)/N1;
cen2 = fc*(e1.^(fmel/ac)-1);

%Mel initialize:
bin_start=zeros(1,number_of_frequency_bands); 
bin_stop=zeros(1,number_of_frequency_bands); 
bin_center=zeros(1,number_of_frequency_bands);

%Mel create:
bin_start(1:number_of_frequency_bands)=cen2(1:number_of_frequency_bands);
bin_stop(1:number_of_frequency_bands)=cen2(2:number_of_frequency_bands+1);

%Mel edge repair:
bin_start = round(bin_start*fft_length/Fs)+1;
bin_stop = round(bin_stop*fft_length/Fs)+1;
bin_start(1)=1;
bin_stop(end)=fft_length/2+1;
bin_center(1:number_of_frequency_bands) = 0.5*(bin_start+bin_stop);
bin_size = bin_stop-bin_start+1;
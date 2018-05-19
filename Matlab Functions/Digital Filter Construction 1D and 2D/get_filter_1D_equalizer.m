function [equalizer_filter,amplitude_gains_linear_full_interpolated] = ...
    get_filter_1D_equalizer(filter_order,Fs,number_of_frequency_bands,maximum_frequency,amplitude_gains_dB,window_name) 
%FILTER ORDER MUST BE EVEN SO FILTER LENGTH MUST BE ODD
 
 
%complete to have nyquist at zero:
frequency_vec = linspace(0,maximum_frequency,number_of_frequency_bands);
frequency_vec_including_nyquist = [frequency_vec(:) ; Fs/2];

%interpolate frequency vec to prepare it for filter creation function:
frequency_vec_full_for_interpolation = linspace(0,Fs/2,floor(filter_order/2)+1);
frequency_vec_full_for_interpolation_normalized = frequency_vec_full_for_interpolation / (Fs/2);

%interpolate gain values to prepare it for filter creation function:
amplitude_gains_linear = 10.^(amplitude_gains_dB/10); %formally should by /20
amplitude_gains_linear_including_nyquist = [amplitude_gains_linear(:) ; 0];
amplitude_gains_linear_full_interpolated = interp1(frequency_vec_including_nyquist , amplitude_gains_linear_including_nyquist , frequency_vec_full_for_interpolation , 'pchip');
amplitude_gains_linear_full_interpolated(end) = 0; %make sure nyquist is zero

%get filter window:
if strcmp(window_name, 'hann')
    filter_window = hann(filter_order+1);
elseif strcmp(window_name, 'hanning')
    filter_window = hanning(filter_order+1);
elseif strcmp(window_name, 'hamming')
    filter_window = hamming(filter_order+1);
end

%actually create the filter using fir2:
filter_ceofficients = fir2(filter_order, frequency_vec_full_for_interpolation_normalized , amplitude_gains_linear_full_interpolated , filter_window);
equalizer_filter = dfilt.dffir(filter_ceofficients);
equalizer_filter.PersistentMemory = true;
equalizer_filter.States = 0;




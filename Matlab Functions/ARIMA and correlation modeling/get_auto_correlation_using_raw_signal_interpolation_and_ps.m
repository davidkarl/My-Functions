function [auto_correlation] = get_auto_correlation_using_raw_signal_interpolation_and_ps(input_signal,number_of_coefficients)
% Computes the first n coefficients of the autocorrelation
% function of the signal y
% c(1) is the correlation at lag 0, c(2) at lag 1 etc.
% The correlation is unbiased and is estimate using FFT
% Here we change the time scale to use the efficient 2^N base FFT
% but there is no point in doing this in MATLAB as it is too slow.

input_signal_length = length(input_signal);
fftsize = 2^ceil(log(input_signal_length)/log(2));
input_signal_linearily_interpolated = zeros(1,fftsize);

%Linearly expand y->q:
for i=1:fftsize
  u = 1 + (i-1)/(fftsize-1)*(input_signal_length-1);
  l = min(floor(u),input_signal_length-1); 
  r = l+1; 
  input_signal_linearily_interpolated(i) = (r-u)*input_signal(l) + (u-l)*input_signal(r);
end

auto_correlation = abs(fft(input_signal_linearily_interpolated)).^2 / fftsize;
auto_correlation = real(ifft(auto_correlation));

%here we should contract the data again:
for i=1:number_of_coefficients
    u = 1 + (i-1)/(input_signal_length-1)*(fftsize-1);
    l = min(floor(u),fftsize-1); 
    r = l+1;
    input_signal_linearily_interpolated(i) = (r-u)*auto_correlation(l) + (u-l)*auto_correlation(r);
end

auto_correlation = input_signal_linearily_interpolated(1:number_of_coefficients)';






















function [auto_correlation] = ...
    get_auto_correlation_for_system_with_inverse_spectrum_of_signal(input_signal,number_of_coefficients)
% Computes the first n coefficients of the autocorrelation
% function of the system with reciprocal spectrum to signal y
% It is used to design whitening filter for y
% c(1) is the correlation at lag 0, c(2) at lag 1 etc.
% The correlation is estimated using FFT

N = length(input_signal);

if N < number_of_coefficients
       error('Signal too short.');
end

auto_correlation = (1 ./ abs(fft(input_signal)).^2 ) / N ;
auto_correlation = real(ifft(auto_correlation));

auto_correlation = auto_correlation(1:number_of_coefficients)';






















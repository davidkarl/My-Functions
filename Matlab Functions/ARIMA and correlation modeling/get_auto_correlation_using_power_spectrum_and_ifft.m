function [auto_correlation_sequence] = get_auto_correlation_using_power_spectrum_and_ifft(input_signal,AR_model_order)
% Computes the first n coefficients of the autocorrelation
% function of the signal y
% c(1) is the correlation at lag 0, c(2) at lag 1 etc.
% The correlation is estimated using FFT

% corrs

N=length(input_signal) ;

auto_correlation_sequence=abs(fft(input_signal)).^2 / N  ;
auto_correlation_sequence=real(ifft(auto_correlation_sequence)) ;
auto_correlation_sequence=auto_correlation_sequence(1:AR_model_order)' ;






















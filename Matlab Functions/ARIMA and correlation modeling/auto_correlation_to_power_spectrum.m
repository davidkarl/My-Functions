function [power_spectrum] = auto_correlation_to_power_spectrum(auto_correlation,number_of_coefficients)
% Calculates power spectrum (square of the amplitude spectrum) 
% of the length n from the correlation function
% Usage: s=c2ps(c,n)

auto_correlation = make_row(auto_correlation); 
auto_correlation_length = length(auto_correlation);
auto_correlation = [auto_correlation zeros(1,number_of_coefficients/2-auto_correlation_length)];
auto_correlation = [auto_correlation auto_correlation(number_of_coefficients/2:-1:1)];

power_spectrum = real(ifft(auto_correlation))*number_of_coefficients^2; % to be consistent with the scaling of fft

power_spectrum = max([power_spectrum ; zeros(1,number_of_coefficients)]); % half wave rectification


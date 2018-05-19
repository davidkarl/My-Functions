function [auto_correlation] = ...
    get_auto_correlation_using_direct_calculation_unbiased(input_signal,number_of_coefficients)
% Computes the first n coefficients of the autocorrelation
% function of the signal y
% c(1) is the correlation at lag 0, c(2) at lag 1 etc.
% The correlation is unbiased

N = length(input_signal);
auto_correlation = zeros(number_of_coefficients,1);

for i=1:number_of_coefficients
    auto_correlation(i) = sum(input_signal(1:N-i+1).*input_signal(i:N))/(N-i+1);
end 




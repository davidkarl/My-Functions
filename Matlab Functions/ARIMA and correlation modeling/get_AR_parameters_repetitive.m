function [AR_parameters] = get_AR_parameters_repetitive(input_signal,AR_model_order) 
% Given a signal x,  this function computes the 'a'
% parameters of an AR synthetising filter of the order n
% It uses normal equations derived from Cayley-Hamilton theorem
% It differs from x2a in that it uses repetitive autocorrelation
% Usage: a=x2ar(x,n)

% The length of the input vector
input_signal_length = length(input_signal);
min_number_of_coefficients = 1000;
number_of_auto_correlation_coefficients = max(2*AR_model_order+1,min_number_of_coefficients);  % We shall use this many coefficients
number_of_auto_correlation_coefficients = min(number_of_auto_correlation_coefficients,input_signal_length);

if (number_of_auto_correlation_coefficients>input_signal_length) 
    error('The signal is too short.');
end 

w = ones(number_of_auto_correlation_coefficients,1) ; %initial weights

number_of_iterations = 7;
for j=1:number_of_iterations
    input_signal = get_auto_correlation_using_power_spectrum_and_ifft(...
                                         input_signal,number_of_auto_correlation_coefficients+1-j);
    
    input_signal = input_signal(2:number_of_auto_correlation_coefficients+1-j); 
    
    input_signal = input_signal/max(abs(input_signal)); % scaling seems necessary
end

AR_parameters = auto_correlation_to_AR_parameters_using_CH_theorem_full(input_signal,AR_model_order);
AR_parameters = stabilize_AR_parameters_by_moving_them_inside_the_unit_circle(AR_parameters);








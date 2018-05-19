function [whitened_filtered_signal] = perform_whitening_by_inverse_AR_filter(input_signal,AR_parameters) 
% This function performs whitening of an input vector y applying inverse
% filter to the AR filter with coefficients a
% Usage: u=arinv(y,a) 

input_signal_length = length(input_signal) ;
AR_model_order = length(AR_parameters) ;

AR_parameters = [-flip(AR_parameters) , 1];
whitened_filtered_signal = zeros(1,input_signal_length);
input_signal = input_signal(:);
AR_parameters=AR_parameters(:);

for i = AR_model_order+1 : input_signal_length
    whitened_filtered_signal(i) = AR_parameters' * input_signal(i-AR_model_order:i);
end

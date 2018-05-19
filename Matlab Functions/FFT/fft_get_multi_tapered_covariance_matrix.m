function R_mt = fft_get_multi_tapered_covariance_matrix( input_signal, covariance_matrix_order , sine_tapers)

input_signal = input_signal(:);
[number_of_sine_tapers , individual_sine_taper_length] = size( sine_tapers );  

% first step is to compute R'= sum_{i=1}^{L}(w_i y)(w_i y)*, which is a
% multitaper method: weight x with each taper w_i, and compute outer
% product, and add them together. 

%Repeat input signal number_of_sine_tapers times:
input_signal_repeated = input_signal( :, ones(1,number_of_sine_tapers)); 

%Weight signal by each of the sine tapers:
input_signal_repeated_weighted_by_sine_tapers = sine_tapers' .* input_signal_repeated; 

%Sum of outer product:
R1 = (input_signal_repeated_weighted_by_sine_tapers * input_signal_repeated_weighted_by_sine_tapers'); 
for k = 1:covariance_matrix_order
    r(k) = sum( diag( R1, k- 1));
end

R_mt = toeplitz( r);  











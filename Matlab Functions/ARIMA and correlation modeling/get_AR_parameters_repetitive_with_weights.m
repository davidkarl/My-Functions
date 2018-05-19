function [AR_parameters] = get_AR_parameters_repetitive_with_weights(input_signal,AR_model_order) 
% Given a signal x, this function uses Cayley-Hamilton theorem to estimate the
% a parameters of an ARX model of the order n that gives the same output x 
% when fed with white noise. x must be at least 2n elements long.
% This routine uses more than 2n autocorr. coefs and does repetitive
% autocorrelation to improve efficiency.
% Usage: a=clhamre(x,n)

% The length of the input vector
input_signal_length = length(input_signal);
min_number_of_coefficients = 1000;
m = max(2*AR_model_order,min_number_of_coefficients) ;  % We shall use this many coefficients
m = min(m,input_signal_length) ;

if (2*AR_model_order>input_signal_length) 
    error('The signal is too short.'); 
end

weights = ones(m,1); %initial weights
xc = input_signal;

number_of_iterations = 7; % Number of iterations
for j = 1:number_of_iterations
    xc = get_auto_correlation_using_direct_calculation(xc,m+1-j); % xc(1)
    xc = xc(2:m+1-j);
    xc = xc/max(abs(xc)); % scaling seems necessary
    
    weights = get_auto_correlation_using_direct_calculation(weights,m+1-j);
    weights = weights(2:m+1-j);
    weights = weights/max(weights);
end  

% Arrange the coefficients into a Toeplitz matrix
number_of_equations_or_degrees_of_freedom = length(xc)-AR_model_order; % the number of equations
number_of_equations_or_degrees_of_freedom = AR_model_order;
C = zeros(number_of_equations_or_degrees_of_freedom,AR_model_order);

for i = 1:number_of_equations_or_degrees_of_freedom 
     C(i,:)=( xc(i:i+AR_model_order-1) ./ (weights(i:i+AR_model_order-1) / weights(i+AR_model_order)) )'; 
end

% Compute the ARX model parameters
AR_parameters = (C\xc(AR_model_order+1:number_of_equations_or_degrees_of_freedom+AR_model_order))';
 
% Revert the a-coefficients into the usual order
AR_parameters = flip(AR_parameters);










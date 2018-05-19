function [AR_parameters] = ...
    auto_correlation_to_AR_parameters_using_CH_theorem_full2(auto_correlation,weights,AR_model_order) 

% Given an autocorrelation function c, this function computes the 'a'
% parameters of an AR synthetising filter of the order n
% It uses normal equations derived from Cayley-Hamilton theorem
% c should begin with autocorrelation at lag 0
% w are the weights associated with corresponding member of c
% It differs from c2ache in that it uses w to remove the bias
% Usage: a=c2ache(c,w,n)


auto_correlation = auto_correlation(:);
weights = weights(:);

%The length of the input vector:
auto_correlation_length = length(auto_correlation);
if (2*AR_model_order>=auto_correlation_length) 
    error('The signal is too short.'); 
end
if (length(weights) ~= auto_correlation_length) 
    error('w and c should have the same length.'); 
end

number_of_equations_or_degrees_of_freedom = auto_correlation_length - AR_model_order - 1;

auto_correlation = auto_correlation(2:auto_correlation_length);
weights = weights(2:auto_correlation_length);

%Arrange the coefficients into a matrix:
C = zeros(number_of_equations_or_degrees_of_freedom,AR_model_order);
for i=1:number_of_equations_or_degrees_of_freedom 
  C(i,:) = (auto_correlation(i:i+AR_model_order-1) ./ (weights(i:i+AR_model_order-1) ...
                        / weights(i+AR_model_order)))'; 
end

%Compute the ARX model parameters:
AR_parameters = (C \ auto_correlation(AR_model_order+1:number_of_equations_or_degrees_of_freedom+AR_model_order))';

%Revert the a-coefficients into the usual order:
AR_parameters = AR_parameters(AR_model_order:-1:1);













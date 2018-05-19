function [AR_parameters] = ...
    auto_correlation_to_AR_parameters_using_CH_theorem_full(auto_correlation,AR_model_order) 

% Given an autocorrelation function c, this function computes the 'a'
% parameters of an AR synthetising filter of the order n
% It uses normal equations derived from Cayley-Hamilton theorem
% c should begin with autocorrelation at lag 0
% It differs from c2ach in that it uses the c vector in its entirety
% Usage: a=c2ache(c,n)


auto_correlation = auto_correlation(:);
auto_correlation_length = length(auto_correlation);

if (2*AR_model_order>=auto_correlation_length) 
    error('The signal is too short.'); 
end

number_of_equations_or_degrees_of_freedom = auto_correlation_length - AR_model_order - 1;

auto_correlation = auto_correlation(2:auto_correlation_length);

%Arrange the coefficients into a matrix:
C = zeros(number_of_equations_or_degrees_of_freedom,AR_model_order);
for i=1:AR_model_order 
    C(:,i) = auto_correlation(i:i+number_of_equations_or_degrees_of_freedom-1);
end

% Compute the ARX model parameters
AR_parameters = (C\auto_correlation(AR_model_order+1:number_of_equations_or_degrees_of_freedom+AR_model_order))';

% Revert the a-coefficients into the usual order:
AR_parameters = AR_parameters(AR_model_order:-1:1);










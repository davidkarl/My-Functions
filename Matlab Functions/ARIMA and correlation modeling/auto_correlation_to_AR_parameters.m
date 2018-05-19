function [AR_parameters] = auto_correlation_to_AR_parameters(auto_correlation_sequence,AR_model_order) 
% Given an autocorrelation function c, this function computes the 'a'
% parameters of an AR synthetising filter of the order n
% It uses the autocorrelation method
% c should begin with autocorrelation at lag 0
% Usage: a=c2a(c,n)

auto_correlation_sequence=auto_correlation_sequence(:) ;
% The length of the input vector
T=length(auto_correlation_sequence) ;
if (2*AR_model_order>=T), error('The signal is too short.') ; end ;


% Arrange the coefficients into a matrix
C=zeros(AR_model_order) ;
for i=1:AR_model_order,
    for j=1:AR_model_order,
        C(j,i)=auto_correlation_sequence(1+abs(j-i)) ;
    end  
end 

% Compute the AR model parameters:
AR_parameters=(C \ auto_correlation_sequence(2:1+AR_model_order))' ;
 





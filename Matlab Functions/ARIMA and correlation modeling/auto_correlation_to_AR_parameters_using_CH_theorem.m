function AR_parameters=auto_correlation_to_AR_parameters_using_CH_theorem(auto_correlation_sequence,AR_model_order) 
% Given an autocorrelation function c, this function computes the 'a'
% parameters of an AR synthetising filter of the order n
% It uses normal equations derived from Cayley-Hamilton theorem
% c should begin with autocorrelation at lag 0
% Usage: a=c2ach(c,n)

%make it a column vector:
auto_correlation_sequence = auto_correlation_sequence(:) ;

%get auto correlation length:
auto_correlation_length = length(auto_correlation_sequence) ;

%check auto correlation received isn't too short to estimate AR model:
if (2*AR_model_order>=auto_correlation_length) 
    error('The signal is too short.'); 
end

%get only part of auto correlation sequence needed (obviously auto_correlation(1)=1):
auto_correlation_sequence = auto_correlation_sequence(2:2*AR_model_order+1);

%Arrange the coefficients into a matrix:
C_auto_correlation_structured_matrx = zeros(AR_model_order);
for i=1:AR_model_order
    C_auto_correlation_structured_matrx(:,i) = auto_correlation_sequence(i:i+AR_model_order-1); 
end 

%Compute the AR model parameters
AR_parameters = (C_auto_correlation_structured_matrx\auto_correlation_sequence(AR_model_order+1:2*AR_model_order))' ;
 
%Reverse the a-coefficients into the usual order
AR_parameters = AR_parameters(AR_model_order:-1:1) ;





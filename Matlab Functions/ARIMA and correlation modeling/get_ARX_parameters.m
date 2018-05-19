function [a, b] = get_ARX_parameters(input_signal,ARX_model_order)
% Identifies ARX model of the n-th order using data y
% uses forward/backward method supplied in ar  - one time identification
% returns vector of n a coefficients and the b coefficient
% Usage: [a b]=idarxb(y,n)

a = th2poly(ar(input_signal,ARX_model_order)) ;
a = -a(2:ARX_model_order+1) ; % ar uses a different conventions than we do

% The following is less accurate than the alternative method 
% c=corrn(y,n+1)' ;
% b=sqrt(sum(c.*[1 -a])) ;

c=sum(input_signal.^2)/length(input_signal) ;
b=sqrt(getnoise(c,a)) ;





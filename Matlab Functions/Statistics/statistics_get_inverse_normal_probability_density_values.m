function x = statistics_get_inverse_normal_probability_density_values(probabilities,mu,variance)
% CSNORMQ Inverse of the univariate normal disribution function.
%
%	X = CSNORMQ(PROB,MU,VAR) Returns the inverse of the univariate
%	normal cumulative distribution function at the values given in
% 	PROB. The parameters of the normal distribution are the mean MU
%	and the variance VAR.
%
% 	See also CSNORMP, CSNORMC, CSEVALNORM

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 



if variance <= 0
   error('Variance must be greater than zero')
   return
end

if ~isempty(find(probabilities<0 | probabilities >1))
   error('Probabilities must be between 0 and 1')
   return
end

sigma=sqrt(variance);
x = sigma*sqrt(2)*erfinv(2*probabilities-1) + mu;

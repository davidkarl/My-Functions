function y = statistics_get_normal_probability_density_at_points(input_vec,mu,variance)
% CSNORMP Univariate normal probability density function.
%
%	Y = CSNORMP(X,MU,VAR) Returns the value of the normal
%	probabilty density function with mean MU and variance
%	VAR at the values given in X.
%
% 	See also CSNORMC, CSNORMQ, CSEVALNORM

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if variance <= 0
   error('Variance must be greater than zero')
   return
end

arg = ((input_vec-mu).^2)/(2*variance);
cons = sqrt(2*pi)*sqrt(variance);

y = (1/cons)*exp(-arg);
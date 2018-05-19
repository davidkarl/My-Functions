function y = statistics_get_gamma_probability_density_in_given_points(values_vec,T_parameter,lambda_parameter)
% CSGAMMP Gamma probability density function.
%
%	Y = CSGAMMP(X,T,LAMBDA) returns the value of
%	the gamma probability density function at the
%	values given in X. The parameters of the gamma
%	distribution are T and LAMBDA.
%
%	The gamma probability density function is given
%	by 
%		{lambda*exp(-lambda*x)(lambda*x)^(t-1)}/Gamma(t)
%
%	See also CSGAMMC, CSGAMPAR, CSGAMRND


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 



if T_parameter <= 0 || lambda_parameter <= 0
   error('Distribution parameters must be greater than zero')
   return
end
y=zeros(size(values_vec));
ind = find(values_vec >= 0);  % pdf defined for these values
y(ind) = lambda_parameter*exp(-lambda_parameter*values_vec(ind)).*(lambda_parameter*values_vec(ind)).^(T_parameter-1);
cons = 1/gamma(T_parameter);
y = cons*y;
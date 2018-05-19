function y = statistics_get_gamma_cumulative_distribution_probability(values_vec,T_parameter,lambda_parameter)
% CSGAMMC Gamma cumulative distribution.
%
%	Y = CSGAMMC(X,T,LAMBDA) Returns the value of the
% 	gamma cumulative distribution function at the
% 	values given in X. The parameters of the gamma
%	distribution are T and LAMBDA.
%
%	The gamma probability density function is given
%	by 
%		{lambda*exp(-lambda*x)(lambda*x)^(t-1)}/Gamma(t)
%
%	See also CSGAMMP, CSGAMPAR, CSGAMRND


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 



if T_parameter <= 0 || lambda_parameter <= 0
   error('Distribution parameters must be greater than zero')
   return
end

y=zeros(size(values_vec));
ind=find(values_vec>=0);
y(ind) = gammainc(lambda_parameter*values_vec(ind),T_parameter);


function [T_parameter,lambda_parameter] = statistics_estimate_gamma_distribution_parameters(input_vec)
% CSGAMPAR Parameter estimation for the gamma distribution.
%
%	[T,LAMBDA] = CSGAMPAR(X) Returns estimates for the parameters
%	T and LAMBDA of the gamma distribution.
%
%	The gamma probability density function is given
%	by 
%		{lambda*exp(-lambda*x)(lambda*x)^(t-1)}/Gamma(t)
%
%	See also CSGAMMP, CSGAMMC, CSGAMRND


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 
% 	Reference: Statistical Theory, 4th Edition, by Bernard
%   Lindgren, page 273



n=length(input_vec);
mu=mean(input_vec);
m2=(1/n)*sum(input_vec.^2);
T_parameter=mu^2/(m2-mu^2);
lambda_parameter = mu/(m2-mu^2);
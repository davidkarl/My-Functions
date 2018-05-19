function y = statistics_get_beta_cumulative_distribution_probability(values_for_cumulative_distribution,alpha_parameter,beta_parameter)
% CSBETAC Univariate beta cumulative distribution function.
%
%   Y = CSBETAC(X,ALPHA,BETA) Returns the value of the
%   cumulative distribution function for the univariate
%   beta distribution with parameters ALPHA and BETA, at
%   the values given in vector X.
%
%   See also CSBETAP, CSBETARND

%   W. L. and A. R. Martinez, 5/22/00
%   Computational Statistics Handbook with MATLAB

if alpha_parameter <= 0 || beta_parameter <= 0
   error('Distribution parameters alpha and beta must be greater than zero')
   return
end
y = zeros(size(values_for_cumulative_distribution));
cons = beta(alpha_parameter,beta_parameter);
ind = find(values_for_cumulative_distribution>=0 & values_for_cumulative_distribution<1);
y(ind) = betainc(values_for_cumulative_distribution(ind),alpha_parameter,beta_parameter); %MATLAB function
y = y/cons;
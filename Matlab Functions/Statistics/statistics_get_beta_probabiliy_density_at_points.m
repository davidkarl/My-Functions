function y = statistics_get_beta_probabiliy_density_at_points(values_to_calculate_probability_at,alpha_parameter,beta_parameter)
% CSBETAP Univariate beta probability density function.
%
%   Y = CSBETAP(X,ALPHA,BETA) Returns the value of the probability
%   density for the univariate beta distribution with
%   parameters ALPHA and BETA, at the values given in the
%   vector X.
%
%   See also CSBETAC, CSBETARND

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


if alpha_parameter <= 0 | beta_parameter <= 0
   error('Distribution parameters alpha and beta must be greater than zero')
   return
end 
y = zeros(size(values_to_calculate_probability_at));
cons = beta(alpha_parameter,beta_parameter);
ind = find(values_to_calculate_probability_at>=0 & values_to_calculate_probability_at<=1);
y(ind) = values_to_calculate_probability_at(ind).^(alpha_parameter-1).*(1-values_to_calculate_probability_at(ind)).^(beta_parameter-1);
y = y/cons;
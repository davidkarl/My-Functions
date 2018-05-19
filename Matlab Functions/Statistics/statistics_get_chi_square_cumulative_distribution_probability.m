function y = statistics_get_chi_square_cumulative_distribution_probability(points_vec,nu_degrees_of_freedom_parameter)
% CSCHIC Chi-square cumulative distribution function.
%
%   Y = CSCHIC(X,NU) Returns the value of the cumulative
%   distribution function for the chi-square distribution
%   with NU degrees of freedom, at a given vector of X values.
%
%   See also CSCHIP, CSCHIRND

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 



if nu_degrees_of_freedom_parameter <= 0
   error('The degrees of freedom must be greater than 0')
   return
end

y=zeros(size(points_vec))
ind = find(points_vec >=0);
y(ind) = gammainc(0.5*points_vec(ind),nu_degrees_of_freedom_parameter/2);
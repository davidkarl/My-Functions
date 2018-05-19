function y = statistics_get_weibull_cumulative_distribution_at_points(x_points,nu,a,b)
% CSWEIBC Weibull cumulative distribution function.
%
%   Y = CSWEIBC(X,NU,ALPHA,BETA) Returns the values
%   of the cumulative distribution function for the
%   Weibull distribution with parameters NU, ALPHA,
%   and BETA at the values given in X.
%
%   See also CSWEIBP, CSWEIBQ


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

if a <= 0 | b <= 0
   error('Distribution parameters alpha and beta must be greater than zero')
   return
end
y=zeros(size(x_points));
ind = find(x_points>nu);
arg=(x_points(ind)-nu)/a;
y(ind) = 1 - exp(-arg.^b);

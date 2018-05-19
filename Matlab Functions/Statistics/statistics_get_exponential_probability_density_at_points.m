function y = statistics_get_exponential_probability_density_at_points(values_vec,lambda_parameter)
% CSEXPOP Exponential probability density function.
%
%   Y = CSEXPOP(X,LAMBDA) Returns the values of the
%   exponential probability density function with
%   parameter LAMBDA, at the given values in X.
%
%   The exponential probability density function is 
%   given by f(x) = lambda*exp(-lambda*x).
%
%   See also CSEXPOC, CSEXPAR, CSEXPOQ, CSEXPRND, CSEXPOPLOT

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if lambda_parameter <= 0
   error('Lambda must be greater than zero')
   return
end
y=zeros(size(values_vec));
ind = find(values_vec>=0);
arg = lambda_parameter*values_vec(ind);
y(ind) = lambda_parameter*exp(-arg);

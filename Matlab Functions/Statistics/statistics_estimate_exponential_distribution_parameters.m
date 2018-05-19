function phat = statistics_estimate_exponential_distribution_parameters(input_vec)
% CSEXPAR Parameter estimation for the exponential distribution.
%
%   LAMBDA = CSEXPAR(X) Returns the parameter estimate for the
%   exponential distribution using the sample given in X.
%
%   The exponential probability density function is 
%   given by f(x) = lambda*exp(-lambda*x).
%
%   See also CSEXPOP, CSEXPOC, CSEXPOQ, CSEXPRND, CSEXPOPLOT

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


phat = 1/mean(input_vec);

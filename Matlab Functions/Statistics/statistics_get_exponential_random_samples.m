function X = statistics_get_exponential_random_samples(number_of_samples,lambda_parameter)
% CSEXPRND Generate exponential random variates.
%
%   X = CSEXPRND(N,LAMBDA) Returns an array of
%   exponentially distributed random variates.
%   This uses the inverse transform method.
%
%   The exponential probability density function is 
%   given by f(x) = lambda*exp(-lambda*x).
%
%   See also CSEXPOP, CSEXPOC, CSEXPOQ, CSEXPAR, CSEXPOPLOT

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

% generate the required uniform random variables
uni = rand(1,number_of_samples);
% transform to exponential
X = -log(uni)/lambda_parameter;

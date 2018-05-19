function y = statistics_get_weibull_probability_density_at_points(x,nu,a,b)
% CSWEIBP Weibull probability density function.
%
%   Y = CSWEIBP(X,NU,ALPHA,BETA) Returns the values
%   of the Weibull probability density function with
%   parameters NU, ALPHA, and BETA at the values 
%   given in X.
%
%   See also CSWEIBC, CSWEIBQ

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if alpha <= 0 | beta <= 0
   error('Distribution parameters alpha and beta must be greater than zero')
   return
end

y=zeros(size(x));
cons=b/a;
ind = find(x>nu);
arg=(x(ins)-nu)/a;
y)ins) = arg.^(b-1).*exp(-arg.^b);
y = cons*y;
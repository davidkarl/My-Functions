function phat = statistics_estimate_poisson_parameter(x)
% CSPOIPAR Parameter estimation for the Poisson distribution.
%
%   PHAT = CSPOIPAR(X) Returns an estimate PHAT for the 
%   parameter lambda of the Poisson distribution using the
%   sample given in X.
%
%   See also CSPOISP, CSPOISC, CSPOIRND, CSPOISSPLOT


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 



lam = mean(x);
function y=smooth2(x,tau)
% SMOOTH2   Simple smoother #2.
%
%    SMOOTH2(X,TAU) returns a smoothed version of vector X, with smoothness
%    controlled by TAU, the response time in samples.  TAU is comparable to
%    the length of a moving average filter in its effect on the smoother's
%    bandwidth.
%
%    SMOOTH2 has perfect response to gradual curvature.
%
%    See also SMOOTH1, LSSMOOTH, IRLSSMOOTH
%

% Written by James S. Montanaro, March 2015

% Compute filter
n=1+2*floor(3.741*tau/2);                  % odd filter length
t=(pi/(n+1))*((1-n):2:(n-1))';             % special linspace
h=.8855+cos(t*(1:3))*[1.684 .9985 .2]';    % filter with flat-top response

% Apply filter (newer MATLAB versions)
y=conv(x,h,'same')./conv(ones(size(x)),h,'same');

% Apply filter (older MATLAB versions)
%y=conv(x,h)./conv(ones(size(x)),h);  y=y((n+1)/2:end-(n-1)/2);

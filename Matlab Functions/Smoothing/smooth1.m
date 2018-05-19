function y=smooth1(x,tau)
% SMOOTH1   Simple smoother #1.
%
%    SMOOTH1(X,TAU) returns a smoothed version of vector X, with smoothness
%    controlled by TAU, the response time in samples.  TAU is comparable to
%    the length of a moving average filter in its effect on the smoother's
%    bandwidth.
%
%    SMOOTH1 has a nonnegative impulse response.
%
%    See also SMOOTH2, LSSMOOTH, IRLSSMOOTH
%

% Written by James S. Montanaro, March 2015

% Compute filter
n=1+2*floor(1.990*tau/2);                  % odd filter length
t=(pi/(n+1))*((1-n):2:(n-1))';             % special linspace
h=.8144+cos(t)+.1856*cos(2*t);             % 3-term filter

% Apply filter (newer MATLAB versions)
y=conv(x,h,'same')./conv(ones(size(x)),h,'same');

% Apply filter (older MATLAB versions)
%y=conv(x,h)./conv(ones(size(x)),h);  y=y((n+1)/2:end-(n-1)/2);

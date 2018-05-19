function [res] = make_2D_ramp(mat_size, direction_in_radians, slope, intercept, origin)
% IM = mkRamp(SIZE, DIRECTION, SLOPE, INTERCEPT, ORIGIN)
%
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of a ramp function, with given gradient DIRECTION
% (radians, CW from X-axis, default = 0), SLOPE (per pixel, default =
% 1), and a value of INTERCEPT (default = 0) at the ORIGIN (default =
% (size+1)/2, [1 1] = upper left).  All but the first argument are
% optional.


mat_size = mat_size(:);
if (size(mat_size,1) == 1)
  mat_size = [mat_size,mat_size];
end

% -----------------------------------------------------------------
% OPTIONAL args:

if ~exist('dir','var')
  direction_in_radians = 0;
end
 
if ~exist('slope','var')
  slope = 1;
end
 
if ~exist('intercept','var')
  intercept = 0;
end

if ~exist('origin','var')
  origin = (mat_size+1)/2;
end

% -----------------------------------------------------------------

xinc = slope*cos(direction_in_radians);
yinc = slope*sin(direction_in_radians);

[xramp,yramp] = meshgrid( xinc*([1:mat_size(2)]-origin(2)), ...
                          yinc*([1:mat_size(1)]-origin(1)) );
 
res = intercept + xramp + yramp;


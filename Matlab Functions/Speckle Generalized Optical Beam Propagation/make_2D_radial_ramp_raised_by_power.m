function [res] = make_2D_radial_ramp_raised_by_power(mat_size, exponential_power, origin)
% IM = mkR(SIZE, EXPT, ORIGIN)
% 
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of a radial ramp function, raised to power EXPT
% (default = 1), with given ORIGIN (default = (size+1)/2, [1 1] =
% upper left).  All but the first argument are optional.

mat_size = mat_size(:);
if (size(mat_size,1) == 1)
  mat_size = [mat_size,mat_size];
end
 
% -----------------------------------------------------------------
% OPTIONAL args:

if ~exist('exponential_power','var')
  exponential_power = 1;
end

if ~exist('origin','var')
  origin = (mat_size+1)/2;
end

% -----------------------------------------------------------------

[xramp,yramp] = meshgrid( [1:mat_size(2)]-origin(2), [1:mat_size(1)]-origin(1) );

res = (xramp.^2 + yramp.^2).^(exponential_power/2);

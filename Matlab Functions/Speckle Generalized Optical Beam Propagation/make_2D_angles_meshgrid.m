function [res] = make_2D_angles_meshgrid(mat_size, phase_offset, origin)
% IM = mkAngle(SIZE, PHASE, ORIGIN)
%
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of the polar angle (in radians, CW from the
% X-axis, ranging from -pi to pi), relative to angle PHASE (default =
% 0), about ORIGIN pixel (default = (size+1)/2).


mat_size = mat_size(:);
if (size(mat_size,1) == 1)
  mat_size = [mat_size,mat_size];
end

% -----------------------------------------------------------------
% OPTIONAL args:

if (exist('origin') ~= 1)
  origin = (mat_size+1)/2;
end

% -----------------------------------------------------------------

[xramp,yramp] = meshgrid( [1:mat_size(2)]-origin(2), [1:mat_size(1)]-origin(1) );

res = atan2(yramp,xramp);

if (exist('phase') == 1)
  res = mod(res+(pi-phase_offset),2*pi)-pi;
end

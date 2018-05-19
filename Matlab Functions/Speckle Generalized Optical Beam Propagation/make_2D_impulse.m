function [res] = make_2D_impulse(mat_size, origin, amplitude)
% IM = mkImpulse(SIZE, ORIGIN, AMPLITUDE)
%
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing a single non-zero entry, at position ORIGIN (defaults to
% ceil(size/2)), of value AMPLITUDE (defaults to 1).

mat_size = mat_size(:)';
if (size(mat_size,2) == 1)
  mat_size = [mat_size mat_size];
end

if (exist('origin') ~= 1)
  origin = ceil(mat_size/2);
end

if (exist('amplitude') ~= 1)
  amplitude = 1;
end

res = zeros(mat_size);
res(origin(1),origin(2)) = amplitude;

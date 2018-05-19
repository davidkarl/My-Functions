function res = reconstruct_laplacian_pyramid(pyramid, index_matrix, levels, filter_vec, boundary_conditions_string)
% RES = reconLpyr(PYR, INDICES, LEVS, FILT2, EDGES)
%
% Reconstruct image from Laplacian pyramid, as created by buildLpyr.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
%
% LEVS (optional) should be a list of levels to include, or the string
% 'all' (default).  The finest scale is number 1.  The lowpass band
% corresponds to lpyrHt(INDICES)+1.
%
% FILT2 (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Default = 'binom5'.  EDGES specifies edge-handling,
% and defaults to 'reflect1' (see corrDn).


if (nargin < 2)
  error('First two arguments (PYR, INDICES) are required');
end
  
%------------------------------------------------------------
% DEFAULTS:

if ~exist('levels','var')
  levels = 'all';
end

if ~exist('filter_vec','var')
  filter_vec = 'binom5';
end

if ~exist('boundary_conditions_string','var')
  boundary_conditions_string= 'reflect1';
end
%------------------------------------------------------------

max_level = 1 + get_laplacian_pyramid_height(index_matrix);
if strcmp(levels,'all')
  levels = [1:max_level]';
else
  if (any(levels > max_level))
    error(sprintf('Level numbers must be in the range [1, %d].', max_level));
  end
  levels = levels(:);
end

if ischar(filter_vec)
  filter_vec = get_filter_by_name(filter_vec);
end

filter_vec = filter_vec(:);
res_sz = index_matrix(1,:);

if any(levels > 1)

  int_sz = [index_matrix(1,1), index_matrix(2,2)];
  
  nres = reconstruct_laplacian_pyramid( pyramid(prod(res_sz)+1:size(pyramid,1)), ...
      index_matrix(2:size(index_matrix,1),:), levels-1, filter_vec, boundary_conditions_string);
  
  if (res_sz(1) == 1)
    res = upsample_inserting_zeros_convolve(...
                                    nres, filter_vec', boundary_conditions_string, [1 2], [1 1], res_sz);
  elseif (res_sz(2) == 1)
    res = upsample_inserting_zeros_convolve(...
                                    nres, filter_vec, boundary_conditions_string, [2 1], [1 1], res_sz);
  else
    hi = upsample_inserting_zeros_convolve(...
                                    nres, filter_vec, boundary_conditions_string, [2 1], [1 1], int_sz);
    res = upsample_inserting_zeros_convolve(...
                                    hi, filter_vec', boundary_conditions_string, [1 2], [1 1], res_sz);
  end

else
  
  res = zeros(res_sz);

end

if any(levels == 1)
  res = res + get_pyramid_subband(pyramid,index_matrix,1);
end

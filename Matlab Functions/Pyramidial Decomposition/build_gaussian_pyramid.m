function [pyramid,index_matrix] = build_gaussian_pyramid(mat_in, pyramid_height, filter_vec, boundary_conditions_string)
% [PYR, INDICES] = buildGpyr(IM, HEIGHT, FILT, EDGES)
%
% Construct a Gaussian pyramid on matrix IM.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is 1+maxPyrHt(size(IM),size(FILT)).
% You can also specify 'auto' to use this value.
%
% FILT (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Default = 'binom5'.  EDGES specifies edge-handling, and
% defaults to 'reflect1' (see corrDn).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.


if (nargin < 1)
    error('First argument (IM) is required');
end

mat_in_size = size(mat_in);

%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('filter_mat','var')
    filter_vec = 'binom5';
end

if ischar(filter_vec)
    filter_vec = get_filter_by_name(filter_vec);
end

if ( (size(filter_vec,1) > 1) && (size(filter_vec,2) > 1) )
    error('FILT should be a 1D filter (i.e., a vector)');
else
    filter_vec = filter_vec(:);
end

max_pyramid_height = 1 + get_max_pyramid_height(mat_in_size, size(filter_vec,1));
if ( ~exist('pyramid_height','var') || strcmp(pyramid_height,'auto') )
    pyramid_height = max_pyramid_height;
else
    if (pyramid_height > max_pyramid_height)
        error(sprintf('Cannot build pyramid higher than %d levels.',max_pyramid_height));
    end
end

if ~exist('boundary_conditions_string','var')
    boundary_conditions_string = 'reflect1';
end

%------------------------------------------------------------

if (pyramid_height <= 1)
    
    pyramid = mat_in(:);
    index_matrix = mat_in_size;
    
else
    
    if (mat_in_size(2) == 1)
        low_passed_next_level = corr2_downsample(mat_in, filter_vec, boundary_conditions_string, [2 1], [1 1]);
    elseif (mat_in_size(1) == 1)
        low_passed_next_level = corr2_downsample(mat_in, filter_vec', boundary_conditions_string, [1 2], [1 1]);
    else
        low_passed_next_level = corr2_downsample(mat_in, filter_vec', boundary_conditions_string, [1 2], [1 1]);
        low_passed_next_level = corr2_downsample(low_passed_next_level, filter_vec, boundary_conditions_string, [2 1], [1 1]);
    end
    
    [next_pyramid_level,index_matrix] = build_gaussian_pyramid(low_passed_next_level, pyramid_height-1, filter_vec, boundary_conditions_string);
    
    %Add on to pyramid structure:
    pyramid = [mat_in(:); next_pyramid_level];
    
    %Add on to index matrix:
    index_matrix = [mat_in_size; index_matrix];
    
end


function [laplacian_pyramid,index_matrix] = build_laplacian_pyramid(mat_in, pyramid_height, filter1, filter2, boundary_conditions_string)
% [PYR, INDICES] = buildLpyr(IM, HEIGHT, FILT1, FILT2, EDGES)
%
% Construct a Laplacian pyramid on matrix (or vector) IM.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is 1+maxPyrHt(size(IM),size(FILT)).  You can also specify 'auto' to
% use this value.
%
% FILT1 (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Default = 'binom5'.  FILT2 specifies the "expansion"
% filter (default = filt1).  EDGES specifies edge-handling, and
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

if ~exist('filter1')
    filter1 = 'binom5';
end

if ischar(filter1)
    filter1 = get_filter_by_name(filter1);
end

if  size(filter1,1) > 1 && size(filter1,2) > 1
    error('FILT1 should be a 1D filter (i.e., a vector)');
else
    filter1 = filter1(:);
end

if ~exist('filt2','var')
    filter2 = filter1;
end

if ischar(filter2)
    filter2 = get_filter_by_name(filter2);
end

if  size(filter2,1) > 1 && size(filter2,2) > 1
    error('FILT2 should be a 1D filter (i.e., a vector)');
else
    filter2 = filter2(:);
end

max_height = 1 + get_max_pyramid_height(mat_in_size, max(size(filter1,1), size(filter2,1)));
if ~exist('pyramid_height','var') || strcmp(pyramid_height,'auto')
    pyramid_height = max_height;
else
    if pyramid_height > max_height
        error(sprintf('Cannot build pyramid higher than %d levels.',max_height));
    end
end

if ~exist('boundary_conditions_string','var')
    boundary_conditions_string= 'reflect1';
end

%------------------------------------------------------------

if (pyramid_height <= 1)
    
    laplacian_pyramid = mat_in(:);
    index_matrix = mat_in_size;
    
else
    
    if (mat_in_size(2) == 1)
        low_passed_next_level = corr2_downsample(mat_in, filter1, boundary_conditions_string, [2 1], [1 1]);
    elseif (mat_in_size(1) == 1)
        low_passed_next_level = corr2_downsample(mat_in, filter1', boundary_conditions_string, [1 2], [1 1]);
    else
        low_passed_next_level = corr2_downsample(mat_in, filter1', boundary_conditions_string, [1 2], [1 1]);
        int_sz = size(low_passed_next_level);
        low_passed_next_level = corr2_downsample(low_passed_next_level, filter1, boundary_conditions_string, [2 1], [1 1]);
    end
    
    [next_pyramid_level,index_matrix] = build_laplacian_pyramid(low_passed_next_level, pyramid_height-1, filter1, filter2, boundary_conditions_string);
    
    %IT SEEMS ODD I HAVE TO FILTER TWICE...SEEMS REDUNDANT SOMEHOW.... WHY
    %NOT SIMPLY TAKE THE BEFORE LOWPASSED IMAGE BEFORE DOWNSAMPLING AND SUBSTRACT IT?
    if (mat_in_size(1) == 1)
        upsampled_and_again_filtered_low_pass = upsample_inserting_zeros_convolve(low_passed_next_level, filter2', boundary_conditions_string, [1 2], [1 1], mat_in_size);
    elseif (mat_in_size(2) == 1)
        upsampled_and_again_filtered_low_pass = upsample_inserting_zeros_convolve(low_passed_next_level, filter2, boundary_conditions_string, [2 1], [1 1], mat_in_size);
    else
        hi = upsample_inserting_zeros_convolve(low_passed_next_level, filter2, boundary_conditions_string, [2 1], [1 1], int_sz);
        upsampled_and_again_filtered_low_pass = upsample_inserting_zeros_convolve(hi, filter2', boundary_conditions_string, [1 2], [1 1], mat_in_size);
    end
    
    %Get the difference between the mat_in and the lowpassed->upsampled->lowpassed image...:
    upsampled_and_again_filtered_low_pass = mat_in - upsampled_and_again_filtered_low_pass;
    
    %I SHOULD PRE-ALLOCATE THIS!!!!!:
    laplacian_pyramid = [upsampled_and_again_filtered_low_pass(:); next_pyramid_level];
    index_matrix = [mat_in_size; index_matrix];
    
end


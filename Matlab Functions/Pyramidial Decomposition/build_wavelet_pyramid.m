function [wavelet_pyramid,index_matrix] = build_wavelet_pyramid(mat_in, height, low_pass_filter_vec, boundary_conditions_string)
% [PYR, INDICES] = buildWpyr(IM, HEIGHT, FILT, EDGES)
%
% Construct a separable orthonormal QMF/wavelet pyramid on matrix (or vector) IM.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is maxPyrHt(IM,FILT).  You can also specify 'auto' to use this value.
%
% FILT (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Filter can be of even or odd length, but should be symmetric. (WHY?!?!?!?)
% Default = 'qmf9'.  EDGES specifies edge-handling, and
% defaults to 'reflect1' (see corrDn).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.


if (nargin < 1)
    error('First argument (IM) is required');
end

%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('filter_vec','var')
    low_pass_filter_vec = 'qmf9';
end

if ~exist('boundary_conditions_string','var')
    boundary_conditions_string= 'reflect1';
end

if ischar(low_pass_filter_vec)
    low_pass_filter_vec = get_filter_by_name(low_pass_filter_vec);
end

if  size(low_pass_filter_vec,1) > 1 && size(low_pass_filter_vec,2) > 1
    error('FILT should be a 1D filter (i.e., a vector)');
else
    low_pass_filter_vec = low_pass_filter_vec(:);
end

%QMF orthogonal filter:
high_pass_filter = modulate_flip_shift(low_pass_filter_vec);

% Stagger sampling if filter is odd-length:
if (mod(size(low_pass_filter_vec,1),2) == 0)
    stag = 2;
else
    stag = 1;
end

mat_in_size = size(mat_in);

max_height = get_max_pyramid_height(mat_in_size, size(low_pass_filter_vec,1));
if  ~exist('height','var') || strcmp(height,'auto')
    height = max_height;
else
    if height > max_height
        error(sprintf('Cannot build pyramid higher than %d levels.',max_height));
    end
end

if (height <= 0)
    
    wavelet_pyramid = mat_in(:);
    index_matrix = mat_in_size;
    
else
    
    if (mat_in_size(2) == 1)
        %mat_in is row vec:
        lolo = corr2_downsample(mat_in, low_pass_filter_vec, boundary_conditions_string, [2 1], [stag 1]);
        hihi = corr2_downsample(mat_in, high_pass_filter, boundary_conditions_string, [2 1], [2 1]);
    elseif (mat_in_size(1) == 1)
        %mat_in is column vec:
        lolo = corr2_downsample(mat_in, low_pass_filter_vec', boundary_conditions_string, [1 2], [1 stag]);
        hihi = corr2_downsample(mat_in, high_pass_filter', boundary_conditions_string, [1 2], [1 2]);
    else
        %mat in is 2D image:
        %WON'T BUILDING A 2D filter from the 1D filter and using corr2_downsampling be faster?
        lo = corr2_downsample(mat_in, low_pass_filter_vec, boundary_conditions_string, [2 1], [stag 1]);
        hi = corr2_downsample(mat_in, high_pass_filter, boundary_conditions_string, [2 1], [2 1]);
        lolo = corr2_downsample(lo, low_pass_filter_vec', boundary_conditions_string, [1 2], [1 stag]);
        lohi = corr2_downsample(hi, low_pass_filter_vec', boundary_conditions_string, [1 2], [1 stag]); % horizontal
        hilo = corr2_downsample(lo, high_pass_filter', boundary_conditions_string, [1 2], [1 2]); % vertical
        hihi = corr2_downsample(hi, high_pass_filter', boundary_conditions_string, [1 2], [1 2]); % diagonal
    end
    
    [next_pyramid_level,index_matrix] = build_wavelet_pyramid(lolo, height-1, low_pass_filter_vec, boundary_conditions_string);
    
    if mat_in_size(1) == 1 || mat_in_size(2) == 1
        wavelet_pyramid = [hihi(:); next_pyramid_level];
        index_matrix = [size(hihi); index_matrix];
    else
        wavelet_pyramid = [lohi(:); hilo(:); hihi(:); next_pyramid_level];
        index_matrix = [size(lohi); size(hilo); size(hihi); index_matrix];
    end
    
end


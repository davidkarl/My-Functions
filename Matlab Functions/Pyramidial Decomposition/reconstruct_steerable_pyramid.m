function res = reconstruct_steerable_pyramid(pyramid, index_matrix, filter_m_filter, boundary_conditions_string, levels, bands)
% RES = reconSpyr(PYR, INDICES, FILTFILE, EDGES, LEVS, BANDS)
%
% Reconstruct image from its steerable pyramid representation, as created
% by buildSpyr.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
%
% FILTFILE (optional) should be a string referring to an m-file that returns
% the rfilters.  examples: sp0Filters, sp1Filters, sp3Filters
% (default = 'sp1Filters').
% EDGES specifies edge-handling, and defaults to 'reflect1' (see
% corrDn).
%
% LEVS (optional) should be a list of levels to include, or the string
% 'all' (default).  0 corresonds to the residual highpass subband.
% 1 corresponds to the finest oriented scale.  The lowpass band
% corresponds to number spyrHt(INDICES)+1.
%
% BANDS (optional) should be a list of bands to include, or the string
% 'all' (default).  1 = vertical, rest proceeding anti-clockwise.


%%------------------------------------------------------------
% DEFAULTS:

if ~exist('filter_m_filter','var')
    filter_m_filter = 'sp1Filters';
end

if ~exist('boundary_conditions_string','var')
    boundary_conditions_string= 'reflect1';
end

if ~exist('levels','var')
    levels = 'all';
end

if ~exist('bands','var')
    bands = 'all';
end

%%------------------------------------------------------------

if ischar(filter_m_filter) && exist(filter_m_filter) == 2
    [lo0filt,hi0filt,lofilt,bfilts,steering_matrix,harmonics] = eval(filter_m_filter);
    number_of_bands = get_steerable_pyramid_number_of_orientation_bands(index_matrix);
    if ((number_of_bands > 0) && (size(bfilts,2) ~= number_of_bands))
        error('Number of pyramid bands is inconsistent with filter file');
    end
else
    error('filtfile argument must be the name of an M-file containing SPYR filters.');
end

max_level =  1 + get_steerable_pyramid_height(index_matrix);
if strcmp(levels,'all')
    levels = [0:max_level]';
else
    if (any(levels > max_level) || any(levels < 0))
        error(sprintf('Level numbers must be in the range [0, %d].', max_level));
    end
    levels = levels(:);
end

if strcmp(bands,'all')
    bands = [1:number_of_bands]';
else
    if (any(bands < 1) || any(bands > number_of_bands))
        error(sprintf('Band numbers must be in the range [1,3].', number_of_bands));
    end
    bands = bands(:);
end

if (get_steerable_pyramid_height(index_matrix) == 0)
    if (any(levels==1))
        res1 = get_pyramid_subband(pyramid,index_matrix,2);
    else
        res1 = zeros(index_matrix(2,:));
    end
else
    res1 = reconstruct_steerable_pyramid_level_recursively(...
                    pyramid(1+prod(index_matrix(1,:)):size(pyramid,1)), ...
                    index_matrix(2:size(index_matrix,1),:), ...
                    lofilt, ...
                    bfilts, ...
                    boundary_conditions_string, ...
                    levels, ...
                    bands);
end

res = upsample_inserting_zeros_convolve(res1, lo0filt, boundary_conditions_string);

% residual highpass subband
if any(levels == 0)
    res = upsample_inserting_zeros_convolve( ...
                    reshape_vec_portion_to_given_dimensions(pyramid, index_matrix(1,:)), ...
                    hi0filt, ...
                    boundary_conditions_string, ...
                    [1 1], ...
                    [1 1], ...
                    size(res), ...
                    res);
end


function [pyramid,index_matrix,steermtx,harmonics] = build_steerable_pyramid(mat_in, height, filter_m_file, boundary_conditions_string)
% [PYR, INDICES, STEERMTX, HARMONICS] = buildSpyr(IM, HEIGHT, FILTFILE, EDGES)
%
% Construct a steerable pyramid on matrix IM.  Convolutions are
% done with spatial filters.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is maxPyrHt(size(IM),size(FILT)).
% You can also specify 'auto' to use this value.
%
% FILTFILE (optional) should be a string referring to an m-file that
% returns the rfilters.  (examples: 'sp0Filters', 'sp1Filters',
% 'sp3Filters','sp5Filters'.  default = 'sp1Filters'). EDGES specifies
% edge-handling, and defaults to 'reflect1' (see corrDn).
% 
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
% See the function STEER for a description of STEERMTX and HARMONICS.

%-----------------------------------------------------------------
% DEFAULTS:

if (exist('filtfile') ~= 1)
    filter_m_file = 'sp1Filters';
end

if ~exist('boundary_conditions_string','var')
    boundary_conditions_string= 'reflect1';
end

if (ischar(filter_m_file) && exist(filter_m_file) == 2)
    [lo0filt,hi0filt,lofilt,bfilts,steermtx,harmonics] = eval(filter_m_file);
else
    fprintf(1,'\nUse buildSFpyr for pyramids with arbitrary numbers of orientation bands.\n');
    error('FILTFILE argument must be the name of an M-file containing SPYR filters.');
end

max_height = get_max_pyramid_height(size(mat_in), size(lofilt,1));
if  ~exist('height','var') || strcmp(height,'auto')
    height = max_height;
else
    if height > max_height
        error(sprintf('Cannot build pyramid higher than %d levels.',max_height));
    end
end 

%-----------------------------------------------------------------

hi0 = corr2_downsample(mat_in, hi0filt, boundary_conditions_string);
lo0 = corr2_downsample(mat_in, lo0filt, boundary_conditions_string);

[pyramid,index_matrix] = construct_steerable_pyramid_level_recursively(...
                                     lo0, height, lofilt, bfilts, boundary_conditions_string);

pyramid = [hi0(:) ; pyramid];
index_matrix = [size(hi0); index_matrix];


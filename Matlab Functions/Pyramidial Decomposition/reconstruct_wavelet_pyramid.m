function res = reconstruct_wavelet_pyramid(pyramid, index_matrix, low_pass_filter, boundary_conditions_string, levels, bands)
% RES = reconWpyr(PYR, INDICES, FILT, EDGES, LEVS, BANDS)
%
% Reconstruct image from its separable orthonormal QMF/wavelet pyramid
% representation, as created by buildWpyr.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
%
% FILT (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Default = 'qmf9'.  EDGES specifies edge-handling,
% and defaults to 'reflect1' (see corrDn).
%
% LEVS (optional) should be a vector of levels to include, or the string
% 'all' (default).  1 corresponds to the finest scale.  The lowpass band
% corresponds to wpyrHt(INDICES)+1.
%
% BANDS (optional) should be a vector of bands to include, or the string
% 'all' (default).   1=horizontal, 2=vertical, 3=diagonal.  This is only used
% for pyramids of 2D images.


if nargin < 2
    error('First two arguments (PYR INDICES) are required');
end

%%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('low_pass_filter','var')
    low_pass_filter = 'qmf9';
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

max_level = 1 + get_wavelet_pyramid_height(index_matrix);
if strcmp(levels,'all')
    levels = [1:max_level]';
else
    if (any(levels > max_level))
        error(sprintf('Level numbers must be in the range [1, %d].', max_level));
    end
    levels = levels(:);
end

if strcmp(bands,'all')
    bands = [1:3]';
else
    if (any(bands < 1) || any(bands > 3))
        error('Band numbers must be in the range [1,3].');
    end
    bands = bands(:);
end

if ischar(low_pass_filter)
    low_pass_filter = get_filter_by_name(low_pass_filter);
end

low_pass_filter = low_pass_filter(:);
high_pass_filter = modulate_flip_shift(low_pass_filter);

%For odd-length filters, stagger the sampling lattices(???):
if (mod(size(low_pass_filter,1),2) == 0)
    stag = 2;
else
    stag = 1;
end

%Compute size of result image: assumes critical sampling (boundaries correct)
res_sz = index_matrix(1,:);
if (res_sz(1) == 1)
    loind = 2;
    res_sz(2) = sum(index_matrix(:,2));
elseif (res_sz(2) == 1)
    loind = 2;
    res_sz(1) = sum(index_matrix(:,1));
else
    loind = 4;
    res_sz = index_matrix(1,:) + index_matrix(2,:);  %%horizontal + vertical bands.
    hres_sz = [index_matrix(1,1), res_sz(2)];
    lres_sz = [index_matrix(2,1), res_sz(2)];
end


%First, recursively collapse coarser scales:
if any(levels > 1)
    
    if size(index_matrix,1) > loind
        nres = reconstruct_wavelet_pyramid( ...
                        pyramid(1+sum(prod(index_matrix(1:loind-1,:)')):size(pyramid,1)), ...
                        index_matrix(loind:size(index_matrix,1),:), ...
                        low_pass_filter, ...
                        boundary_conditions_string, ...
                        levels-1, ...
                        bands);
    else
        nres = get_pyramid_subband(pyramid, index_matrix, loind); 	% lowpass subband
    end
    
    if (res_sz(1) == 1)
        res = upsample_inserting_zeros_convolve(...
                                nres, low_pass_filter', boundary_conditions_string, [1 2], [1 stag], res_sz);
    elseif (res_sz(2) == 1)
        res = upsample_inserting_zeros_convolve(...
                                nres, low_pass_filter, boundary_conditions_string, [2 1], [stag 1], res_sz);
    else
        ires = upsample_inserting_zeros_convolve(...
                                nres, low_pass_filter', boundary_conditions_string, [1 2], [1 stag], lres_sz);
        res = upsample_inserting_zeros_convolve(...
                                ires, low_pass_filter, boundary_conditions_string, [2 1], [stag 1], res_sz);
    end
    
else
    
    res = zeros(res_sz);
    
end


% Add in reconstructed bands from this level:
if any(levels == 1)
    if (res_sz(1) == 1)
        upsample_inserting_zeros_convolve(...
            get_pyramid_subband(pyramid,index_matrix,1), ...
            high_pass_filter', ...
            boundary_conditions_string, ...
            [1 2], ...
            [1 2], ...
            res_sz, ...
            res);
    
    elseif (res_sz(2) == 1)
        upsample_inserting_zeros_convolve(...
            get_pyramid_subband(pyramid,index_matrix,1), ...
            high_pass_filter, ...
            boundary_conditions_string, ...
            [2 1], ...
            [2 1], ...
            res_sz, ...
            res);
    
    else
        if any(bands == 1) % horizontal
            ires = upsample_inserting_zeros_convolve(...
                get_pyramid_subband(pyramid,index_matrix,1),...
                low_pass_filter',...
                boundary_conditions_string,...
                [1 2],...
                [1 stag],...
                hres_sz);
            
            %destructively modify res
            upsample_inserting_zeros_convolve(...
                ires,...
                high_pass_filter,...
                boundary_conditions_string,...
                [2 1],...
                [2 1],...
                res_sz,...
                res);  
        end
        if any(bands == 2) % vertical
            ires = upsample_inserting_zeros_convolve(...
                get_pyramid_subband(pyramid,index_matrix,2),...
                high_pass_filter',...
                boundary_conditions_string,...
                [1 2],...
                [1 2],...
                lres_sz);
            
            %destructively modify res
            upsample_inserting_zeros_convolve(...
                ires,...
                low_pass_filter,...
                boundary_conditions_string,...
                [2 1],...
                [stag 1],...
                res_sz,...
                res); 
        end
        if any(bands == 3) % diagonal
            ires = upsample_inserting_zeros_convolve(...
                get_pyramid_subband(pyramid,index_matrix,3),...
                high_pass_filter',...
                boundary_conditions_string,...
                [1 2],...
                [1 2],...
                hres_sz);
            
            %destructively modify res
            upsample_inserting_zeros_convolve(...
                ires,...
                high_pass_filter,...
                boundary_conditions_string,...
                [2 1],...
                [2 1],...
                res_sz,...
                res);
        end
    end
end


function res = reconstruct_steerable_pyramid_fourier(pyramid, index_matrix, levels, bands, transition_width)
% RES = reconSFpyr(PYR, INDICES, LEVS, BANDS, TWIDTH)
%
% Reconstruct image from its steerable pyramid representation, in the Fourier
% domain, as created by buildSFpyr.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
%
% LEVS (optional) should be a list of levels to include, or the string
% 'all' (default).  0 corresonds to the residual highpass subband.
% 1 corresponds to the finest oriented scale.  The lowpass band
% corresponds to number spyrHt(INDICES)+1.
%
% BANDS (optional) should be a list of bands to include, or the string
% 'all' (default).  1 = vertical, rest proceeding anti-clockwise.
%
% TWIDTH is the width of the transition region of the radial lowpass
% function, in octaves (default = 1, which gives a raised cosine for
% the bandpass filters).

%%% MODIFIED VERSION, 7/04, uses different lookup table for radial frequency!

%%------------------------------------------------------------
% DEFAULTS:

if ~exist('levels','var')
    levels = 'all';
end

if ~exist('bands','var')
    bands = 'all';
end

if ~exist('transition_width','var')
    transition_width = 1;
elseif (transition_width <= 0)
    fprintf(1,'Warning: TWIDTH must be positive.  Setting to 1.\n');
    transition_width = 1;
end

%%------------------------------------------------------------

nbands = get_steerable_pyramid_number_of_orientation_bands(index_matrix);

max_levels =  1 + get_steerable_pyramid_height(index_matrix);
if strcmp(levels,'all')
    levels = [0:max_levels]';
else
    if (any(levels > max_levels) || any(levels < 0))
        error(sprintf('Level numbers must be in the range [0, %d].', max_levels));
    end
    levels = levels(:);
end

if strcmp(bands,'all')
    bands = [1:nbands]';
else
    if (any(bands < 1) || any(bands > nbands))
        error(sprintf('Band numbers must be in the range [1,3].', nbands));
    end
    bands = bands(:);
end

%----------------------------------------------------------------------

dims = index_matrix(1,:);
ctr = ceil((dims+0.5)/2);

[xramp,yramp] = meshgrid( ([1:dims(2)]-ctr(2))./(dims(2)/2), ...
                          ([1:dims(1)]-ctr(1))./(dims(1)/2) );
angle = atan2(yramp,xramp);
log_rad = sqrt(xramp.^2 + yramp.^2);
log_rad(ctr(1),ctr(2)) =  log_rad(ctr(1),ctr(2)-1);
log_rad  = log2(log_rad);

% Radial transition function (a raised cosine in log-frequency):
[Xrcos,Yrcos] = make_raised_cosine(transition_width,(-transition_width/2),[0 1]);
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(abs(1.0 - Yrcos.^2));

if (size(index_matrix,1) == 2)
    if (any(levels==1))
        resdft = fftshift(fft2(get_pyramid_subband(pyramid,index_matrix,2)));
    else
        resdft = zeros(index_matrix(2,:));
    end
else
    resdft = reconstruct_steerable_pyramid_fourier_level_recursively(...
                        pyramid(1+prod(index_matrix(1,:)):size(pyramid,1)), ...
                        index_matrix(2:size(index_matrix,1),:), ...
                        log_rad, Xrcos, Yrcos, angle, nbands, levels, bands);
end

lo0mask = apply_point_operation_to_image(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
resdft = resdft .* lo0mask;

% residual highpass subband
if any(levels == 0)
    hi0mask = apply_point_operation_to_image(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
    hidft = fftshift(fft2(reshape_vec_portion_to_given_dimensions(pyramid, index_matrix(1,:))));
    resdft = resdft + hidft .* hi0mask;
end

res = real(ifft2(ifftshift(resdft)));

function [pyramid,index_matrix,steering_matrix,harmonics] = build_seperable_pyramid_fourier_domain(...
    mat_in, height, order, transition_width)
% [PYR, INDICES, STEERMTX, HARMONICS] = buildSFpyr(IM, HEIGHT, ORDER, TWIDTH)
%
% Construct a steerable pyramid on matrix IM, in the Fourier domain.
% This is similar to buildSpyr, except that:
%
%    + Reconstruction is exact (within floating point errors) WHY IS IT NOT WITH THE REGULAR SEPERABLE PYRAMID
%    + It can produce any number of orientation bands.
%    - Typically slower, especially for non-power-of-two sizes.
%    - Boundary-handling is circular.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is maxPyrHt(size(IM),size(FILT));
%
% The squared radial functions tile the Fourier plane, with a raised-cosine
% falloff.  Angular functions are cos(theta-k\pi/(K+1))^K, where K is
% the ORDER (one less than the number of orientation bands, default= 3).
%
% TWIDTH is the width of the transition region of the radial lowpass
% function, in octaves (default = 1, which gives a raised cosine for
% the bandpass filters).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
% See the function STEER for a description of STEERMTX and HARMONICS.

%-----------------------------------------------------------------
% DEFAULTS:

max_height = floor(log2(min(size(mat_in)))) - 2;

if ~exist('height','var')
    height = max_height;
else
    if height > max_height
        error(sprintf('Cannot build pyramid higher than %d levels.',max_height));
    end
end

if ~exist('order','var')
    order = 3;
elseif order > 15 || order < 0
    fprintf(1,'Warning: ORDER must be an integer in the range [0,15]. Truncating.\n');
    order = min(max(order,0),15);
else
    order = round(order);
end
number_of_bands = order+1;

if ~exist('transition_width','var')
    transition_width = 1;
elseif (transition_width <= 0)
    fprintf(1,'Warning: TWIDTH must be positive.  Setting to 1.\n');
    transition_width = 1;
end

%-----------------------------------------------------------------
% Steering stuff:

if (mod((number_of_bands),2) == 0)
    harmonics = [0:(number_of_bands/2)-1]'*2 + 1;
else
    harmonics = [0:(number_of_bands-1)/2]'*2;
end

steering_matrix = get_steering_matrix_from_harmonic(harmonics, pi*[0:number_of_bands-1]/number_of_bands, 'even');

%-----------------------------------------------------------------

mat_in_size = size(mat_in);
mat_in_centers = ceil((mat_in_size+0.5)/2);

[xramp,yramp] = meshgrid( ([1:mat_in_size(2)]-mat_in_centers(2))./(mat_in_size(2)/2), ...
                          ([1:mat_in_size(1)]-mat_in_centers(1))./(mat_in_size(1)/2) );
angle = atan2(yramp,xramp);
log_rad = sqrt(xramp.^2 + yramp.^2);
log_rad(mat_in_centers(1),mat_in_centers(2)) =  log_rad(mat_in_centers(1),mat_in_centers(2)-1); %?
log_rad  = log2(log_rad);

%Radial transition function (a raised cosine in log-frequency):
[Xrcos,Yrcos] = make_raised_cosine(transition_width,(-transition_width/2),[0 1]);
Yrcos = sqrt(Yrcos);

YIrcos = sqrt(1.0 - Yrcos.^2); 
lo0mask = apply_point_operation_to_image(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
mat_in_fft = fftshift(fft2(mat_in));
lo0dft =  mat_in_fft .* lo0mask;

[pyramid,index_matrix] = construct_steerable_pyramid_fourier_level_recursively(...
                                    lo0dft, log_rad, Xrcos, Yrcos, angle, height, number_of_bands);

hi0mask = apply_point_operation_to_image(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
hi0dft =  mat_in_fft .* hi0mask;
hi0 = ifft2(ifftshift(hi0dft));

pyramid = [real(hi0(:)) ; pyramid];
index_matrix = [size(hi0); index_matrix];

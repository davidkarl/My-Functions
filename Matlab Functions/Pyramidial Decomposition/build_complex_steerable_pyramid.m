function [pyramid,index_matrix,steering_matrix,harmonics] = build_complex_steerable_pyramid(mat_in, height, order, transition_width)
% [PYR, INDICES, STEERMTX, HARMONICS] = buildSCFpyr(IM, HEIGHT, ORDER, TWIDTH)
%
% This is a modified version of buildSFpyr, that constructs a
% complex-valued steerable pyramid  using Hilbert-transform pairs
% of filters.  Note that the imaginary parts will *not* be steerable.
%
% To reconstruct from this representation, either call reconSFpyr
% on the real part of the pyramid, *or* call reconSCFpyr which will
% use both real and imaginary parts (forcing analyticity).
%
% Description of this transform appears in: Portilla & Simoncelli,
% Int'l Journal of Computer Vision, 40(1):49-71, Oct 2000.
% Further information: http://www.cns.nyu.edu/~eero/STEERPYR/

% Modified by Javier Portilla to return complex (quadrature pair) channels,

 
%-----------------------------------------------------------------
% DEFAULTS:

max_ht = floor(log2(min(size(mat_in)))) - 2;

if ~exist('height','var')
    height = max_ht;
else
    if height > max_ht
        error(sprintf('Cannot build pyramid higher than %d levels.',max_ht));
    end
end

if ~exist('order','var')
    order = 3;
elseif order > 15  || order < 0
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
mat_in_center = ceil((mat_in_size+0.5)/2);

[xramp,yramp] = meshgrid( ([1:mat_in_size(2)]-mat_in_center(2))./(mat_in_size(2)/2), ...
                          ([1:mat_in_size(1)]-mat_in_center(1))./(mat_in_size(1)/2) );
angle = atan2(yramp,xramp);
log_rad = sqrt(xramp.^2 + yramp.^2); 
log_rad(mat_in_center(1),mat_in_center(2)) =  log_rad(mat_in_center(1),mat_in_center(2)-1);
log_rad  = log2(log_rad);

% Radial transition function (a raised cosine in log-frequency):
[Xrcos,Yrcos] = make_raised_cosine(transition_width,(-transition_width/2),[0 1]);
Yrcos = sqrt(Yrcos);

YIrcos = sqrt(1.0 - Yrcos.^2);
lo0mask = apply_point_operation_to_image(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
mat_in_fft = fftshift(fft2(mat_in));
lo0dft =  mat_in_fft .* lo0mask;

[pyramid,index_matrix] = construct_complex_steerable_pyramid_level_recursively(lo0dft, log_rad, Xrcos, Yrcos, angle, height, number_of_bands);

hi0mask = apply_point_operation_to_image(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
hi0dft =  mat_in_fft .* hi0mask;
hi0 = ifft2(ifftshift(hi0dft));

pyramid = [real(hi0(:)) ; pyramid];
index_matrix = [size(hi0); index_matrix];

function mat_in_amplitude_equalized = test_riesz_monogenic_analysis_with_amplitude_equalization(mat_in, ...
                                                                                                number_of_scales, ...
                                                                                                smoothing_filter_sigma)
%DEMO_MONOGENICANALYSIS_AMPLITUDEQUALIZATION equalize the monogenic
%amplitude
% AEQ = DEMO_MONOGENICANALYSIS_AMPLITUDEQUALIZATION(A, NUMSCALES, SIGMA)
% perform monogenic analysis with smoothing SIGMA for NUMSCALES scales of
% the image A, equalize the monogenic amplitudes, and reconstruct the image
% AEQ.
%
% --------------------------------------------------------------------------
% Input arguments:
%
% They are all optional. 
%
% A: input image. If none is provided the method buildDefaultImage is
% called to build one.
%
% NUMSCALES: number of wavelet scales for the decomposition. Default is
% 3.
%
% SIGMA: standard deviation of the Gaussian window that is used for
% regularizing the monogenic analysis. Default is 1.5;
%
% --------------------------------------------------------------------------
%
% Output arguments:
%
% AEQ: image that is synthetized after equalizing the monogenic amplitudes
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

%create a default image:
mat_in = double(imread('barbara.tif'));
mat_in = mat_in(1:256,1:256);
figure;
imagesc(mat_in);
axis image;
axis off;
colormap gray;
title('original image')


%setup the 2D Riesz transform:
number_of_scales = 3;
smoothing_filter_sigma = 1.5;
flag_restrict_angle_values = 0;

%do full monogenic analysis:
[~, ~, amplitude_cell_array, phase_cell_array] = riesz_full_monogenic_analysis(mat_in,  ...
                                                                               number_of_scales, ...
                                                                               smoothing_filter_sigma, ...
                                                                               flag_restrict_angle_values);
 
%% display phase and amplitude images
for scale_counter = 1:number_of_scales,
    figure;
    imagesc(amplitude_cell_array{scale_counter});
    colormap gray;
    axis image;
    axis off;
    title(sprintf('Amplitude at scale %d', scale_counter));
    figure;
    imagesc(phase_cell_array{scale_counter});
    colormap gray;
    axis image; 
    axis off;
    title(sprintf('Phase at scale %d', scale_counter));
end

%% equalize the amplitude without changing the phase
% configure the Riesz transform of order 1:
riesz_transform_object1 = riesz_transform_object(size(mat_in), 1, number_of_scales, 1);
[mat_in_riesz_wavelet_cell_array , highpass_residual] = multiscale_riesz_analysis(mat_in, riesz_transform_object1);

%rescale coefficients to account for amplitude equalization:
minimum_quantile_to_discard = 0.1;
for scale_counter = 1:number_of_scales,
    map = amplitude_cell_array{scale_counter};
    threshold = quantile(amplitude_cell_array{scale_counter}(:), minimum_quantile_to_discard); % discard low amplitude coefficients
    mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,1) = ...
                                (mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,1)./map).*(map>threshold);
    mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,2) = ...
                                (mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,2)./map).*(map>threshold);
end
mat_in_riesz_wavelet_cell_array{number_of_scales+1} = zeros(size(mat_in_riesz_wavelet_cell_array{number_of_scales+1}));
highpass_residual = zeros(size(highpass_residual));

%reconstruct:
mat_in_amplitude_equalized = multiscale_riesz_synthesis(mat_in_riesz_wavelet_cell_array, highpass_residual, riesz_transform_object1);
figure;
imagesc(mat_in_amplitude_equalized),
colormap(gray);
axis image;
axis off;
title('Amplitude equalized image');
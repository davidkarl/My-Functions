function [rotation_angles_cell_array , coherency_cell_array] = test_riesz_monogenic_analysis(mat_in, ...
                                                                                             number_of_scales, ...
                                                                                             smoothing_filter_sigma, ...
                                                                                             flag_restrict_angle_values)
%DEMO_MONOGENICANALYSIS(A, NUMSCALES, SIGMA, FULL) perform monogenic
%analysis
%
% [ANG COHERENCY] = DEMO_MONOGENICANALYSIS(A, NUMSCALES, SIGMA, FULL)
% perform the monogenic analysis described in [1] for the input image A.
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
% FULL: evaluate the angles over the full range [-pi, pi]. Default is 0.
% This yields angles in the range [-pi/2, pi/2]
%
% --------------------------------------------------------------------------
%
% Output arguments:
%
% ANG: angles estimated by the monogenic analysis in the wavelet bands
%
% COHERENCY: estimated coherency in the wavelet bands
%
% --------------------------------------------------------------------------
%
% References:
%
% [1] Multiresolution Monogenic Signal Analysis Using the Riesz-Laplace Wavelet Transform
% M. Unser, D. Sage, D. Van De Ville. IEEE Transactions on Image Processing, vol. 18, no. 11,
% pp. 2402-2418, November 2009.
%


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

%do most basic monogenic analysis:
[rotation_angles_cell_array , coherency_cell_array] = ...
            riesz_monogenic_analysis(mat_in, number_of_scales, smoothing_filter_sigma, flag_restrict_angle_values);

%% display angles and coherency at each scale
for j = 1:number_of_scales,
    %display angles
    figure;
    imagesc(rotation_angles_cell_array{j});
    colormap('hsv');
    axis off;
    axis image;
    title(sprintf('Angle at scale %d', j));
    %display coherency
    figure;
    imagesc(coherency_cell_array{j});
    colormap('gray');
    axis off;
    axis image;
    title(sprintf('Coherency at scale %d', j));
    %display a composite image
    figure;
    clear hsv;
    hsv = zeros(size(rotation_angles_cell_array{j}, 1), size(rotation_angles_cell_array{j}, 2), 3);
    if flag_restrict_angle_values == 1
        hsv(:,:,1) = (rotation_angles_cell_array{j}+pi)/(2*pi);
    else
        hsv(:,:,1) = (rotation_angles_cell_array{j}+pi/2)/pi;
    end
    hsv(:,:,2) = coherency_cell_array{j};
    hsv(:,:,3) = ones(size(hsv,1), size(hsv,2));
    rgb = hsv2rgb(hsv);
    imagesc(rgb)
    axis off;
    axis image;
    title(sprintf('Composite scale and coherency at scale %d', j));
end
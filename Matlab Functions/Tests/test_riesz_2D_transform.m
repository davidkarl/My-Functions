% function [Q R] = test_riesz_2D_transform(mat_in, N)
%RIESZ2DDEMO 2D Riesz-wavelet transform decomposition and reconstruction
%
%  [Q R] = DEMO_RIESZ2D() decompose a default image in the Riesz-wavelet
%  frame of order 2 and reconstruct it from the coefficients.
%
%  [Q R] = DEMO_RIESZ2D(A) decompose the image A in the Riesz-wavelet
%  frame of order 2 and reconstruct it from the coefficients.
%
%  [Q R] = DEMO_RIESZ2D(A, N) decompose the image A in the Riesz-wavelet
%  frame of order N and reconstruct it from the coefficients.


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
riesz_transform_order = 2;
number_of_wavelet_scales = 3;

%setup the Riesz tranform parameters and pre compute the Riesz filters:
riesz_transform_object1 = riesz_transform_object(size(mat_in), riesz_transform_order, number_of_wavelet_scales, 1);
 
%visualize the Riesz filters in frequency domain:
display('Showing Riesz filters');
visualize_riesz_filters(riesz_transform_object1);

%compute the Riesz-wavelet coefficients:
[riesz_wavelet_matrices_cell_array,mat_in_highpassed] = multiscale_riesz_analysis(mat_in,riesz_transform_object1);

%Visualize coefficients:
display('Showing Riesz-wavelet coefficients')
visualize_riesz_2D_wavelet_coefficients(riesz_transform_object1, riesz_wavelet_matrices_cell_array)



%Synthesis step:
mat_in_reconstructed = multiscale_riesz_synthesis(riesz_wavelet_matrices_cell_array, mat_in_highpassed, riesz_transform_object1);

%Check for perfect reconstruction:
fprintf('Maximum absolute value reconstruction error: %e\n',max(abs(double(mat_in(:))-mat_in_reconstructed(:))));
fprintf('Root mean square error: %e\n',sqrt(mean(abs(double(mat_in(:))-mat_in_reconstructed(:)).^2)));
figure; 
imagesc(mat_in); 
axis image, 
axis off, 
colormap gray;
title('reconstructed image')


% end
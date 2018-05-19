% A script to show how the new pyramid is used
% Copyright, Neal Wadhwa, August 2014
%
% Part of the Supplementary Material to:
%
% Riesz Pyramids for Fast Phase-Based Video Magnification
% Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
% Computational Photography (ICCP), 2014 IEEE International Conference on


%Create an impulse image:
impulse_mat = zeros(128,128,'single');
impulse_mat(65,65) = 1;

%Build the pyramid and show the impulse and frequency responses:
[riesz_pyramid, index_matrix] = build_riesz_pyramid(impulse_mat);


L = size(index_matrix,1);
figure('Position', [1 1 1000 400]);
for k = 1:L
    subplot(2,L,k);
    imagesc(get_pyramid_subband(riesz_pyramid,index_matrix,k));
    ylabel('Space');
    xlabel('Space');
    title(sprintf('Level %d', k));
end
for k = 1:L 
    subplot(2,L,L+k);
    imagesc(abs(fftshift(fft2(get_pyramid_subband(riesz_pyramid,index_matrix,k)))));
    ylabel('Frequency');
    xlabel('Frequency');
end
% fprintf('Press any key to continue\n');
% pause; 

% Collapse the pyramid and compare to original
reconstructed = reconstruct_riesz_pyramid(riesz_pyramid, index_matrix);

figure();
subplot(1,2,1);
imshow(impulse_mat);
title('Original');
subplot(1,2,2);
imshow(reconstructed);
title('Reconstructed');
fprintf('The error is %0.2fdB\n', -10*log10(mean((impulse_mat(:)-reconstructed(:)).^2)));
% fprintf('Press any key to continue\n');
% pause;




%%
% Load a test image built into matlab
image_mat = im2single(imread('cameraman.tif'));

% Build a new pyramid and display it
[riesz_pyramid, index_matrix] = build_riesz_pyramid(image_mat);

% Show the levels
L = size(index_matrix,1);
figure('Position', [1 1 1200 300]); 
for k = 1:L
    image_subband = get_pyramid_subband(riesz_pyramid,index_matrix,k);
    [orientation, phase, amplitude] = riesz_transform_for_image_subband(image_subband);
    subplot(1,L,k);
%     imagesc(image_subband);
    imagesc(phase);
    ylabel('Space');
    xlabel('Space');
    title(sprintf('Level %d', k));
end
% fprintf('Press any key to continue\n');
% pause;


% Show reconstruction
reconstructed = reconstruct_riesz_pyramid(riesz_pyramid,index_matrix);
figure();
subplot(1,2,1); 
imshow(image_mat);
title('Original');
subplot(1,2,2);
imshow(reconstructed);
title('Reconstructed');
fprintf('The error is %0.2fdB\n', -10*log10(mean((image_mat(:)-reconstructed(:)).^2)));



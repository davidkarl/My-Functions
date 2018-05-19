% DENOISEDEMO   Denoise demo
% Compare the denoise performance of wavelet and contourlet transforms
%

mat_in = double(imread('lena.png'));

%Parameters:
noise_variance_vec = [5];
addpath('C:\Users\master\Desktop\matlab\WAVELETS AND PYRAMIDS\contourlet_toolbox');


for noise_variance_counter = 1:length(noise_variance_vec)

    %Generate noisy image:
    noise_variance = noise_variance_vec(noise_variance_counter); 
    noise = randn(size(mat_in));
    noise_normalized = noise/std(noise(:));
    noise = noise_variance .* noise_normalized;
    mat_in_noisy = mat_in + noise;
    resnoi(noise_variance_counter) = get_PSNR(mat_in, mat_in_noisy);
    
    %(1).
    %%%%% Wavelet denoising %%%%%
    %Wavelet transform using PDFB with zero number of levels for DFB:
    laplacian_pyramid_filter_name = 'db5';
    diamond_filter_name = 'db5';
    threshold_in_terms_of_sigma = 3; % lead to 3*sigma threshold denoising
    %Do Pyramidial direction filter bank decomposition:
    y = PDFB_decomposition(mat_in_noisy, laplacian_pyramid_filter_name, diamond_filter_name, [0 0 0]);
    [c, s] = PDFB_to_vec(y);
    %Threshold (typically 3*sigma):
    wavelet_threshold = threshold_in_terms_of_sigma * noise_variance;
    c = c .* (abs(c) > wavelet_threshold);
    %Reconstruction:
    y = vec_to_PDFB(c, s);
    denoised_pyramid = PDFB_reconstruction(y, laplacian_pyramid_filter_name, diamond_filter_name);
    reswav(noise_variance_counter) = get_PSNR(mat_in, denoised_pyramid);
    
    
    %(2).
    %%%%% PDTDFB Denoising %%%%%
    cfg =  [2 3 4 5];
    residual = false;
    y = PDTDFB_decomposition(mat_in_noisy, [1 4 5], 'nalias', 'meyer', 'db5', residual);
    %threshold for PDTDFB is lower because the energy of the filter is not 1:
    thd = 0.6*wavelet_threshold;
    yth = PDTDFB_threshold(y,'threshold', [0 thd thd wavelet_threshold], cfg); %cfg not initialized at first
    pim = PDTDFB_reconstruction(yth, 'nalias', 'meyer', 'db5', residual);
    %Decomposte frequency domain:
    cfg =  [2 3 4 5];
    frequency_windows = get_PDTDFB_frequency_windows(512, 0.1, length(cfg));
    y = PDTDFB_decomposition_FFT(mat_in_noisy, cfg, frequency_windows);
    yth = PDTDFB_threshold(y,'threshold', [0 thd thd thd wavelet_threshold], cfg);
    pim = PDTDFB_reconstruction_FFT(yth, frequency_windows);
    respdtdfb(noise_variance_counter) = get_PSNR(mat_in, pim);

    
    %(3).
    %%%%% Contourlet Denoising %%%%%
    %Contourlet transform:
    nlevs = [0 4 5];
    laplacian_pyramid_filter_name = 'db5';
    diamond_filter_name = 'pkva';
    y = PDFB_decomposition(mat_in_noisy, laplacian_pyramid_filter_name, diamond_filter_name, nlevs);
    [c, s] = PDFB_to_vec(y);
    %Threshold:
    %Require to estimate the noise standard deviation in the PDFB domain first
    %since PDFB is not an orthogonal transform
    nvar = PDFB_noise_distribution_estimate(size(mat_in,1), size(mat_in, 2), laplacian_pyramid_filter_name, diamond_filter_name, nlevs);
    cth = threshold_in_terms_of_sigma * noise_variance * sqrt(nvar);
    %Slightly different thresholds for the finest scale:
    fs = s(end, 1);
    fssize = sum(prod(s(find(s(:, 1) == fs), 3:4), 2));
    cth(end-fssize+1:end) = (4/3) * cth(end-fssize+1:end);
    c = c .* (abs(c) > cth);
    %Reconstruction:
    y = vec_to_PDFB(c, s);
    cim = PDFB_reconstruction(y, laplacian_pyramid_filter_name, diamond_filter_name);
    rescon(noise_variance_counter) = get_PSNR(mat_in, cim);

end

%%%%% Plot: Only the hat!
range = [0, 255];
colormap gray;
subplot(2,3,1); 
imagesc(mat_in(41:168, 181:308), range); 
axis image off
set(gca, 'FontSize', 8);
title('Original Image', 'FontSize', 10);

subplot(2,3,2); 
imagesc(mat_in_noisy(41:168, 181:308), range); 
axis image off
set(gca, 'FontSize', 8);
title(sprintf('Noisy Image (PSNR = %.2f dB)', get_PSNR(mat_in, mat_in_noisy)), 'FontSize', 10);

subplot(2,3,4); 
imagesc(denoised_pyramid(41:168, 181:308), range); 
axis image off
set(gca, 'FontSize', 8);
title(sprintf('Denoise using Wavelets (PSNR = %.2f dB)', get_PSNR(mat_in, denoised_pyramid)), 'FontSize', 10);

subplot(2,3,5); 
imagesc(cim(41:168, 181:308), range); 
axis image off
set(gca, 'FontSize', 8);
title(sprintf('Denoise using Contourlets (PSNR = %.2f dB)', get_PSNR(mat_in, cim)), 'FontSize', 10);
          
subplot(2,3,6); 
imagesc(pim(41:168, 181:308), range); 
axis image off
set(gca, 'FontSize', 8);
title(sprintf('Denoise using PDTDFB (PSNR = %.2f dB)', get_PSNR(mat_in, pim)), 'FontSize', 10);


figure;          
colormap gray;
subplot(2,2,1); 
imagesc(mat_in, range); 
axis image off
set(gca, 'FontSize', 8);
title('Original Image', 'FontSize', 10);

subplot(2,2,2); 
imagesc(mat_in_noisy, range); 
axis image off
set(gca, 'FontSize', 8);
title(sprintf('Noisy Image (SNR = %.2f dB)', get_SNR(mat_in, mat_in_noisy)), 'FontSize', 10);

subplot(2,2,3); 
imagesc(denoised_pyramid, range); 
axis image off
set(gca, 'FontSize', 8);
title(sprintf('Denoise using Wavelets (SNR = %.2f dB)', get_SNR(mat_in, denoised_pyramid)), 'FontSize', 10);

subplot(2,2,4); 
imagesc(cim, range); 
axis image off
set(gca, 'FontSize', 8);
title(sprintf('Denoise using Contourlets (SNR = %.2f dB)', get_SNR(mat_in, cim)), 'FontSize', 10);          
          
          
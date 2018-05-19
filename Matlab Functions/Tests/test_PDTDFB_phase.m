% denoising evalution of different curvelet/contourlet implementation
close all; clear all;

im = double(imread('lena512.bmp'));
s = 512;

sigma = 5; % NV(in);
noise = randn(size(im));
noise = sigma.*(noise./(std(noise(:))));

% cfg =  [4 5 5 6];
%cfg =  [3 4 4 5];
cfg =  [2 2 2 2 2];
alpha = 0.15;
s = 512;
resi = false;

frequency_windows = get_PDTDFB_frequency_windows(s, alpha, length(cfg), resi);

% 
disp('Compute all thresholds');
Ff = ones(s);
X = fftshift(ifft2(Ff)) * sqrt(prod(size(Ff)));

yn = PDTDFB_decomposition_FFT(noise, cfg, frequency_windows, alpha, resi);
y = PDTDFB_decomposition_FFT(im, cfg, frequency_windows, alpha, resi);
clear eng


for scale = 1:length(cfg)
    for dir = 1:2^cfg(scale)
        Y_coef_real = y{scale+1}{1}{dir};
        % imaginary part
        Y_coef_imag = y{scale+1}{2}{dir};
        % Signal variance estimation
        Y_coef = Y_coef_real+j*Y_coef_imag;
        
        Y_noise_real = yn{scale+1}{1}{dir};
        % imaginary part
        Y_noise_imag = yn{scale+1}{2}{dir};
        % Signal variance estimation
        Y_noise = Y_noise_real+j*Y_noise_imag;
        
        Y_coef = abs(Y_noise).*Y_coef./abs(Y_coef);

        y{scale+1}{1}{dir} = real(Y_coef);
        y{scale+1}{2}{dir} = imag(Y_coef);

    end
end
y{1} = zeros(size(y{1}));

% Inverse Transform
tic
pim = PDTDFB_reconstruction_FFT(y, frequency_windows, alpha,resi);toc
respdtdfb = get_PSNR(im, pim);




% ------------------------------------------------------------------------

cfg =  [3 3 4];
alpha = 0.15;
s = 512;
resi = false;

frequency_windows = get_PDTDFB_frequency_windows(s, alpha, length(cfg), resi);

%
disp('Compute all thresholds');
Ff = ones(s);
X = fftshift(ifft2(Ff)) * sqrt(prod(size(Ff)));

yn = PDTDFB_decomposition_FFT(noise, cfg, frequency_windows, alpha, resi);
y = PDTDFB_decomposition_FFT(im, cfg, frequency_windows, alpha, resi);

scale = 3; 
dir = 1;

for dir = 1:2^cfg(scale)
    if dir < 2^cfg(scale-1)+1 
        sh = [0 1];
    else
        sh = [1 0];
    end 
    
    Y_coef_real = y{scale+1}{1}{dir};
    % imaginary part
    Y_coef_imag = y{scale+1}{2}{dir};
    % Signal variance estimation
    Y_coef = Y_coef_real+j*Y_coef_imag;

    Y_coefs = circshift(Y_coef,sh);

    diff_ang = angle(Y_coefs./Y_coef);
    

    subplot(121);hist(diff_ang(:),30);

    Y_coef_real = yn{scale+1}{1}{dir};
    % imaginary part
    Y_coef_imag = yn{scale+1}{2}{dir};
    % Signal variance estimation
    Y_coef = Y_coef_real+j*Y_coef_imag;

    Y_coefs = circshift(Y_coef,sh);

    diff_ang = angle(Y_coefs./Y_coef);

    subplot(122);hist(diff_ang(:),30);
    
    pause;
end






%
disp('Testing the two function PDTDFBDEC_F and PDTDFBREC_F')

close all; clear all;

s = 1024;
s = [s s];
cf = [4 4 4];
res = false;

im = make_2D_zone_plate(s);
tic
F = get_PDTDFB_frequency_windows(s, 0.1, length(cf), res );
disp('Window calculation')
toc
tic
y2 = PDTDFB_decomposition_FFT(im, cf, F, [], res);
disp('Foward transform')
toc

tic
imr = PDTDFB_reconstruction_FFT(y2,F,[],res);
disp('Inverse transform')
toc

%
disp('Signal to noise ratio') 
get_SNR(im,imr)

% existing : 23 and 25 sec

for in = 1:2^(cf(1))
    yz = get_PDTDFB_structure_with_zeros(s, cf);

    szb = size(yz{2}{1}{in});

    yz{2}{1}{in}(szb(1)/2+1, szb(2)/2+1) = 1;
    imr = PDTDFB_reconstruction_FFT(yz,F);

    yz = get_PDTDFB_structure_with_zeros(s, cf);

    yz{2}{2}{in}(szb(1)/2+1, szb(2)/2+1) = 1;
    imi = PDTDFB_reconstruction_FFT(yz,F);

%     figure(1);
%     subplot(121);
%     imagesc(fit_matrix_dimensions_to_certain_size(imr,20),0.1*[-1 1]);
%     subplot(122);
%     imagesc(fit_matrix_dimensions_to_certain_size(imi,20),0.1*[-1 1]);
%     colormap gray

    figure;
    imagesc(abs(fftshift(fft2(imr+1j*imi))));
    
%     pause;
end

% =====================================================================
% generate figure for paper part 1, figure 9
% =====================================================================
s = 256;

im = make_2D_impulse(s);
res = true;
cf = 3;
tic
F = get_PDTDFB_frequency_windows(s, 0.3, length(cf), res );
disp('Window calculation')
toc

tic
% y = pdtdfbdec_f(im, cf, F, [], res);
dfilt = 'meyer';
pfilt = 'meyer';
y = PDTDFB_decomposition(im, cf, pfilt,dfilt,[], res);
disp('Foward transform')
toc

yz = get_PDTDFB_structure_with_zeros(256, 3, res);
yz{2}{1}{1} = y{2}{1}{1};
% im1 = pdtdfbrec_f(yz, F, [], res);
im1 = PDTDFB_reconstruction(yz, pfilt,dfilt,[], res);

yz = get_PDTDFB_structure_with_zeros(256, 3, res);
yz{2}{2}{1} = y{2}{2}{1};
% im2 = pdtdfbrec_f(yz, F, [], res);
im2 = PDTDFB_reconstruction(yz, pfilt,dfilt,[], res);

im3 = im1 + im2;

subplot(131); freqz2(fit_matrix_dimensions_to_certain_size(im1,64));
subplot(132); freqz2(fit_matrix_dimensions_to_certain_size(im2,64));
subplot(133); freqz2(fit_matrix_dimensions_to_certain_size(im3,64));


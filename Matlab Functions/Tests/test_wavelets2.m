test_wavelets2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1 - Basic filters, upsampling and downsampling.
N = 1024;
f_vec = (-N/2:N/2-1) / (N/2);
% Low pass filter.
low_pass_filter_decomposition = [0.5, 0.5];
low_pass_filter_fft = fftshift(fft(low_pass_filter_decomposition,N));
plot(f_vec,abs(low_pass_filter_fft))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Frequency response of Haar lowpass filter: [1/2 1/2]')
% High pass filter.
high_pass_filter_decomposition = [0.5, -0.5];
high_pass_filter_fft = fftshift(fft(high_pass_filter_decomposition,N));
plot(f_vec,abs(high_pass_filter_fft))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Frequency response of Haar highpass filter [1/2 -1/2]')
% Linear interpolating lowpass filter.
low_pass_interpolation_filter = [0.5 1 0.5];
low_pass_interpolation_filter_fft = fftshift(fft(low_pass_interpolation_filter,N));
plot(f_vec,abs(low_pass_interpolation_filter_fft))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Frequency response of lowpass filter [1/2 1 1/2]')
% Upsampling.
upsampling_filter = [0.5 0 0.5 0];
upsampling_filter_fft = fftshift(fft(upsampling_filter,N));
plot(f_vec,abs(upsampling_filter_fft))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Fourier transform of [1/2 0 1/2 0]')
% Downsampling.
downsampling_filter = [-1 0 9 16 9 0 -1] / 16;
downsampling_filter_fft = fftshift(fft(downsampling_filter,N));
plot(f_vec,abs(downsampling_filter_fft))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Fourier transform of x = [-1 0 9 16 9 0 -1] / 16')

downsampling_filter2 = [-1 9 9 -1] / 16;
downsampling_filter2_fft = fftshift(fft(downsampling_filter2,N));
plot(f_vec,abs(downsampling_filter2_fft))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Fourier transform of [-1 9 9 -1] / 16')

%WHY IS THE SWITCH FROM X(w) to X(w/2) shown here correct??????:
XX = fftshift(fft(downsampling_filter,2*N));   % X(w)
XX2 = XX(N/2+1 : 3*N/2);       % X(w/2)
XXPi = fftshift(XX);         % X(w+pi)
XX2Pi = XXPi(N/2+1 : 3*N/2);   % X(w/2+pi)
Y = (XX2 + XX2Pi) / 2;
plot(f_vec,abs(Y))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('[X(\omega/2) + X(\omega/2+pi)]/2')




%Product filter examples (what are product filters?!?!?!?!?!):
p = 2;
switch p
case 1,
  % Degree 2
  b = [1 2 1];  % (1 + z^-1)^2
  q = 1 / 2;
  p0 = [1 2 1] / 2;   % conv(b, q)

case 2,
  % Degree 6
  b = [1 4 6 4 1];  % (1 + z^-1)^4
  q = [-1 4 -1] / 16;
  p0 = [-1 0 9 16 9 0 -1] / 16;  % conv(b, q)

case 3,
  % Degree 10
  b = [1 6 15 20 15 6 1];   % (1 + z^-1)^6
  q = [3 -18 38 -18 3] / 256;
  p0 = [3 0 -25 0 150 256 150 0 -25 0 3] / 256;  % conv(b,q)

case 4,
  % Degree 14
  b = [1 8 28 56 70 56 28 8 1];  % (1 + z^-1)^8
  q = [-5 40 -131 208 -131 40 -5] / 2048;
  p0 = [-5 0 49 0 -245 0 1225 2048 1225 0 -245 0 49 0 -5] / 2048;  % conv(b,q)

otherwise,
  % Degree 4p-2
  [p0,b,q] = prodfilt(p);
end
zplane(p0);
title(sprintf('Zeros of the product filter with degree %d', 4*p-2))

[P,f_vec] = dtft(p0,512);
plot(f_vec/pi, abs(P))
xlabel('Angular frequency (normalized by pi)')
ylabel('Frequency response magnitude')
title(sprintf('Frequency response of the product filter with degree %d', 4*p-2))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-D signal analysis

% biorwavf generates symmetric biorthogonal wavelet filters.
% The argument has the form biorNr.Nd, where
%    Nr = number of zeros at pi in the synthesis lowpass filter, s[n].
%    Nd = number of zeros at pi in the analysis lowpass filter, a[n].
% We find the famous Daubechies 9/7 pair, which have Nr = Nd = 4.  
% Note: In earlier versions of the Matlab Wavelet Toolbox (v2.1 and 
% below,) the vectors s and a are zero-padded to make their lengths equal:
%    a[-4] a[-3] a[-2] a[-1] a[0] a[1] a[2] a[3] a[4]
%      0   s[-3] s[-2] s[-1] s[0] s[1] s[2] s[3]   0
%BIORWAVF Biorthogonal spline wavelet filters.
%   [RF,DF] = BIORWAVF(W) returns two SCALING filters
%   associated with the biorthogonal wavelet specified
%   by the string W.
[reconstruction_filter_scaling,decomposition_filter_scaling] = biorwavf('bior4.4');  

% Find the zeros and plot them.
close all
clf
fprintf(1,'Zeros of H0(z)')
roots(decomposition_filter_scaling)
subplot(1,2,1)
zplane(decomposition_filter_scaling)
title('Zeros of H0(z)')

fprintf(1,'Zeros of F0(z)')
roots(reconstruction_filter_scaling)
subplot(1,2,2)
zplane(reconstruction_filter_scaling)          % Note: there are actually 4 zeros clustered at z = -1.
title('Zeros of F0(z)')



% Determine the complete set of filters, with proper alignment.
% Note: Matlab uses the convention that a[n] is the flip of h0[n].
%    h0[n] = flip of a[n], with the sum normalized to sqrt(2).
%    f0[n] = s[n], with the sum normalized to sqrt(2).
%    h1[n] = f0[n], with alternating signs reversed (starting with the first.)
%    f1[n] = h0[n], with alternating signs reversed (starting with the second.)
[low_pass_filter_decomposition,high_pass_filter_decomposition,low_pass_filter_reconstruction,high_pass_filter_reconstruction] = ...
                                                                    biorfilt(decomposition_filter_scaling, reconstruction_filter_scaling);

clf
subplot(2,2,1)
stem(0:8,low_pass_filter_decomposition(2:10))
ylabel('h0[n]')
xlabel('n')
subplot(2,2,2)
stem(0:6,low_pass_filter_reconstruction(2:8))
ylabel('f0[n]')
xlabel('n')
v = axis; axis([v(1) 8 v(3) v(4)])
subplot(2,2,3)
stem(0:6,high_pass_filter_decomposition(2:8))
ylabel('h1[n]')
xlabel('n')
v = axis; axis([v(1) 8 v(3) v(4)])
subplot(2,2,4)
stem(0:8,high_pass_filter_reconstruction(2:10))
ylabel('f1[n]')
xlabel('n')



%Examine the Frequency response of the filters:
N = 512;
f_vec = 2/N*(-N/2:N/2-1);
low_pass_filter_fft = fftshift(fft(low_pass_filter_decomposition,N));
high_pass_filter_fft = fftshift(fft(high_pass_filter_decomposition,N));
F0 = fftshift(fft(low_pass_filter_reconstruction,N));
F1 = fftshift(fft(high_pass_filter_reconstruction,N));
clf
plot(f_vec, abs(low_pass_filter_fft), '-', f_vec, abs(high_pass_filter_fft), '--', f_vec, abs(F0), '-.', f_vec, abs(F1), ':')
title('Frequency responses of Daubechies 9/7 filters')
xlabel('Angular frequency (normalized by pi)')
ylabel('Frequency response magnitude')
legend('H0', 'H1', 'F0', 'F1', 0)



% Load a test signal. 
load noisdopp

noisy_doppler_signal = noisdopp;
levels_vec = length(noisy_doppler_signal);
clear noisdopp

% Compute the lowpass and highpass coefficients using convolution and
% downsampling.
low_pass_level1_coefficients = dyaddown(conv(noisy_doppler_signal,low_pass_filter_decomposition));
high_pass_level1_coefficients = dyaddown(conv(noisy_doppler_signal,high_pass_filter_decomposition));

% The function dwt provides a direct way to get the same result.
[low_pass_level1_coefficients2,high_pass_level1_coefficients2] = dwt(noisy_doppler_signal,'bior4.4');

% Now, reconstruct the signal using upsamping and convolution.  We only
% keep the middle L coefficients of the reconstructed signal i.e. the ones
% that correspond to the original signal.
level1_approximation = conv( dyadup(low_pass_level1_coefficients) , low_pass_filter_reconstruction ) + ...
       conv( dyadup(high_pass_level1_coefficients) , high_pass_filter_reconstruction );
level1_approximation = wkeep(level1_approximation,levels_vec);

% The function idwt provides a direct way to get the same result.
level1_approximation2 = idwt(low_pass_level1_coefficients,high_pass_level1_coefficients,'bior4.4');

% Plot the results.
subplot(4,1,1);
plot(noisy_doppler_signal)
axis([0 1024 -12 12])
title('Single stage wavelet decomposition')
ylabel('x')
subplot(4,1,2);
plot(low_pass_level1_coefficients)
axis([0 1024 -12 12])
ylabel('y0')
subplot(4,1,3);
plot(high_pass_level1_coefficients)
axis([0 1024 -12 12])
ylabel('y1')
subplot(4,1,4);
plot(level1_approximation)
axis([0 1024 -12 12])
ylabel('xhat')


% Next, we perform a three level decomposition.  The following
% code draws the structure of the iterated analysis filter bank.
clf
t_vec = wtree(noisy_doppler_signal,3,'bior4.4');
plot(t_vec)

close(2)

% For a multilevel decomposition, we use wavedec instead of dwt.
% Here we do 3 levels.  wc is the vector of wavelet transform
% coefficients.  l is a vector of lengths that describes the
% structure of wc.
[wavelet_coefficients_mat,levels_vec] = wavedec(noisy_doppler_signal,3,'bior4.4');

% We now need to extract the lowpass coefficients and the various
% highpass coefficients from wc.
signal_approximation_coefficients_level3 = appcoef(wavelet_coefficients_mat,levels_vec,'bior4.4',3);
signal_details_coefficients_level3 = detcoef(wavelet_coefficients_mat,levels_vec,3); %why is it that with the detail coefficients don't need the wavelet name???
signal_details_coefficients_level2 = detcoef(wavelet_coefficients_mat,levels_vec,2);
signal_details_coefficients_level1 = detcoef(wavelet_coefficients_mat,levels_vec,1);

clf
subplot(5,1,1)
plot(noisy_doppler_signal)
axis([0 1024 -22 22])
ylabel('x')
title('Three stage wavelet decomposition')
subplot(5,1,2)
plot(signal_approximation_coefficients_level3)
axis([0 1024 -22 22])
ylabel('a3')
subplot(5,1,3)
plot(signal_details_coefficients_level3)
axis([0 1024 -22 22])
ylabel('d3')
subplot(5,1,4)
plot(signal_details_coefficients_level2)
axis([0 1024 -22 22])
ylabel('d2')
subplot(5,1,5)
plot(signal_details_coefficients_level1)
axis([0 1024 -22 22])
ylabel('d1')


% We can reconstruct each branch of the tree separately from the individual
% vectors of transform coefficients using upcoef.
%MAKE SURE THAT ra3 etc' equal signal_lowpass_approximation_level3 etc'
ra3 = upcoef('a',signal_approximation_coefficients_level3,'bior4.4',3,1024);
rd3 = upcoef('d',signal_details_coefficients_level3,'bior4.4',3,1024);
rd2 = upcoef('d',signal_details_coefficients_level2,'bior4.4',2,1024);
rd1 = upcoef('d',signal_details_coefficients_level1,'bior4.4',1,1024);

% The sum of these reconstructed branches gives the full recontructed signal.
level1_approximation = ra3 + rd3 + rd2 + rd1;

clf
subplot(5,1,1)
plot(noisy_doppler_signal)
axis([0 1024 -10 10])
ylabel('x')
title('Individually reconstructed branches')
subplot(5,1,2)
plot(ra3)
axis([0 1024 -10 10])
ylabel('ra3')
subplot(5,1,3)
plot(rd3)
axis([0 1024 -10 10])
ylabel('rd3')
subplot(5,1,4)
plot(rd2)
axis([0 1024 -10 10])
ylabel('rd2')
subplot(5,1,5)
plot(rd1)
axis([0 1024 -10 10])
ylabel('rd1')


clf
plot(level1_approximation-noisy_doppler_signal)
title('Reconstruction error (using upcoef)')
axis tight


% We can also reconstruct individual branches from the full vector of
% transform coefficients, wc.
rra3 = wrcoef('a',wavelet_coefficients_mat,levels_vec,'bior4.4',3);
rrd3 = wrcoef('d',wavelet_coefficients_mat,levels_vec,'bior4.4',3);
rrd2 = wrcoef('d',wavelet_coefficients_mat,levels_vec,'bior4.4',2);
rrd1 = wrcoef('d',wavelet_coefficients_mat,levels_vec,'bior4.4',1);
level1_approximation2 = rra3 + rrd3 + rrd2 + rrd1;

clf
plot(level1_approximation2-noisy_doppler_signal)
title('Reconstruction error (using wrcoef)')
axis tight


% To reconstruct all branches at once, use waverec.
xxxhat = waverec(wavelet_coefficients_mat,levels_vec,'bior4.4');

clf
plot(xxxhat-noisy_doppler_signal)
axis tight
title('Reconstruction error (using waverec)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 2D image analysis.

% Load a test image.  Matlab test images consist of a matrix, X,
% color palette, map, which maps each value of the matrix to a
% color.  Here, we will apply the Discrete Wavelet Transform to X.
load woman2
%load detfingr; X = X(1:200,51:250);
woman_image = X;
close all
clf
image(woman_image)
colormap(map)
axis image; 
set(gca,'XTick',[],'YTick',[]);
title('Original');


% We will use the 9/7 filters with symmetric extension at the
% boundaries.
dwtmode('sym')
wavelet_name = 'bior4.4';

%Plot the structure of a two stage filter bank.
t_vec = wtree(woman_image,2,'bior4.4');
plot(t_vec)

% Compute a 2-level decomposition of the image using the 9/7 filters.
[wavelet_coefficients_mat,index_matrix] = wavedec2(woman_image,2,wavelet_name);

% Extract the level 1 coefficients.
approximation_coefficients_level1 = appcoef2(wavelet_coefficients_mat,index_matrix,wavelet_name,1);         
detail_coefficients_horizontal_level1 = detcoef2('h',wavelet_coefficients_mat,index_matrix,1);           
detail_coefficients_vertical_level1 = detcoef2('v',wavelet_coefficients_mat,index_matrix,1);           
detail_coefficients_diagonal_level1 = detcoef2('d',wavelet_coefficients_mat,index_matrix,1);           

% Extract the level 2 coefficients.
approximation_coefficients_level2 = appcoef2(wavelet_coefficients_mat,index_matrix,wavelet_name,2);
detail_coefficients_horizontal_level2 = detcoef2('h',wavelet_coefficients_mat,index_matrix,2);
detail_coefficients_vertical_level2 = detcoef2('v',wavelet_coefficients_mat,index_matrix,2);
detail_coefficients_diagonal_level2 = detcoef2('d',wavelet_coefficients_mat,index_matrix,2);

% Display the decomposition up to level 1 only.
number_of_columns = size(map,1);              % Number of colors.
image_size_vec = size(woman_image);
cod_a1 = wcodemat(approximation_coefficients_level1,number_of_columns);  %WHY DO I KEEP ONLY THE INNER HALF IMAGE ONLY?!?!?!?
cod_a1 = wkeep(cod_a1, image_size_vec/2);
cod_h1 = wcodemat(detail_coefficients_horizontal_level1,number_of_columns); 
cod_h1 = wkeep(cod_h1, image_size_vec/2);
cod_v1 = wcodemat(detail_coefficients_vertical_level1,number_of_columns); 
cod_v1 = wkeep(cod_v1, image_size_vec/2);
cod_d1 = wcodemat(detail_coefficients_diagonal_level1,number_of_columns); 
cod_d1 = wkeep(cod_d1, image_size_vec/2);

image([cod_a1,cod_h1;cod_v1,cod_d1]);
axis image; 
set(gca,'XTick',[],'YTick',[]); 
title('Single stage decomposition')
colormap(map)


% Display the entire decomposition upto level 2.
cod_a2 = wcodemat(approximation_coefficients_level2,number_of_columns); 
cod_a2 = wkeep(cod_a2, image_size_vec/4);
cod_h2 = wcodemat(detail_coefficients_horizontal_level2,number_of_columns); 
cod_h2 = wkeep(cod_h2, image_size_vec/4);
cod_v2 = wcodemat(detail_coefficients_vertical_level2,number_of_columns); 
cod_v2 = wkeep(cod_v2, image_size_vec/4);
cod_d2 = wcodemat(detail_coefficients_diagonal_level2,number_of_columns); 
cod_d2 = wkeep(cod_d2, image_size_vec/4);
image([[cod_a2,cod_h2;cod_v2,cod_d2],cod_h1;cod_v1,cod_d1]);
axis image; 
set(gca,'XTick',[],'YTick',[]); 
title('Two stage decomposition')
colormap(map)


% Here are the reconstructed branches
reconstructed_approximation_level2 = wrcoef2('a',wavelet_coefficients_mat,index_matrix,wavelet_name,2);
reconstructed_detail_horizontal_level2 = wrcoef2('h',wavelet_coefficients_mat,index_matrix,wavelet_name,2);
reconstructed_detail_vertical_level2 = wrcoef2('v',wavelet_coefficients_mat,index_matrix,wavelet_name,2);
reconstructed_detail_diagonal_level2 = wrcoef2('d',wavelet_coefficients_mat,index_matrix,wavelet_name,2);

reconstructed_approximation_level1 = wrcoef2('a',wavelet_coefficients_mat,index_matrix,wavelet_name,1);
reconstructed_detail_horizontal_level1 = wrcoef2('h',wavelet_coefficients_mat,index_matrix,wavelet_name,1);
reconstructed_detail_vertical_level1 = wrcoef2('v',wavelet_coefficients_mat,index_matrix,wavelet_name,1);
reconstructed_detail_diagonal_level1 = wrcoef2('d',wavelet_coefficients_mat,index_matrix,wavelet_name,1);

cod_ra2 = wcodemat(reconstructed_approximation_level2,number_of_columns);
cod_rh2 = wcodemat(reconstructed_detail_horizontal_level2,number_of_columns);
cod_rv2 = wcodemat(reconstructed_detail_vertical_level2,number_of_columns);
cod_rd2 = wcodemat(reconstructed_detail_diagonal_level2,number_of_columns);
cod_ra1 = wcodemat(reconstructed_approximation_level1,number_of_columns);
cod_rh1 = wcodemat(reconstructed_detail_horizontal_level1,number_of_columns);
cod_rv1 = wcodemat(reconstructed_detail_vertical_level1,number_of_columns);
cod_rd1 = wcodemat(reconstructed_detail_diagonal_level1,number_of_columns);
subplot(3,4,1); image(woman_image); axis image; set(gca,'XTick',[],'YTick',[]); title('Original')
subplot(3,4,5); image(cod_ra1); axis image; set(gca,'XTick',[],'YTick',[]); title('ra1')
subplot(3,4,6); image(cod_rh1); axis image; set(gca,'XTick',[],'YTick',[]); title('rh1')
subplot(3,4,7); image(cod_rv1); axis image; set(gca,'XTick',[],'YTick',[]); title('rv1')
subplot(3,4,8); image(cod_rd1); axis image; set(gca,'XTick',[],'YTick',[]); title('rd1')
subplot(3,4,9); image(cod_ra2); axis image; set(gca,'XTick',[],'YTick',[]); title('ra2')
subplot(3,4,10); image(cod_rh2); axis image; set(gca,'XTick',[],'YTick',[]); title('rh2')
subplot(3,4,11); image(cod_rv2); axis image; set(gca,'XTick',[],'YTick',[]); title('rv2')
subplot(3,4,12); image(cod_rd2); axis image; set(gca,'XTick',[],'YTick',[]); title('rd2')


% Adding together the reconstructed average at level 2 and all of
% the reconstructed details gives the full reconstructed image.
Xhat = reconstructed_approximation_level2 + reconstructed_detail_horizontal_level2 + reconstructed_detail_vertical_level2 + reconstructed_detail_diagonal_level2 + reconstructed_detail_horizontal_level1 + reconstructed_detail_vertical_level1 + reconstructed_detail_diagonal_level1;
sprintf('Reconstruction error (using wrcoef2) = %g', max(max(abs(woman_image-Xhat))))

% Another way to reconstruct the image.
XXhat = waverec2(wavelet_coefficients_mat,index_matrix,wavelet_name);
sprintf('Reconstruction error (using waverec2) = %g', max(max(abs(woman_image-XXhat))))

% Compression can be accomplished by applying a threshold to the
% wavelet coefficients.  wdencmp is the function that does this.
% 'h' means use hard thresholding. Last argument = 1 means do not
% threshold the approximation coefficients.
%    perfL2 = energy recovery = 100 * ||wc_comp||^2 / ||wc||^2.
%             ||.|| is the L2 vector norm.
%    perf0 = compression performance = Percentage of zeros in wc_comp.
threshold = 20;                                                    
[X_comp,wc_comp,s_comp,perf0,perfL2] = wdencmp('gbl',wavelet_coefficients_mat,index_matrix,wavelet_name,2,threshold,'h',1);

clf
subplot(1,2,1); image(woman_image); axis image; set(gca,'XTick',[],'YTick',[]);
title('Original')
cod_X_comp = wcodemat(X_comp,number_of_columns);
subplot(1,2,2); image(cod_X_comp); axis image; set(gca,'XTick',[],'YTick',[]);
title('Compressed using global hard threshold')
xlabel(sprintf('Energy retained = %2.1f%% \nNull coefficients = %2.1f%%',perfL2,perf0))


% Better compression can be often be obtained if different thresholds
% are allowed for different subbands.
thr_h = [21 17];        % horizontal thresholds.              
thr_d = [23 19];        % diagonal thresholds.                
thr_v = [21 17];        % vertical thresholds.                
threshold = [thr_h; thr_d; thr_v];
[X_comp,wc_comp,s_comp,perf0,perfL2] = wdencmp('lvd',woman_image,wavelet_name,2,threshold,'h');

clf
subplot(1,2,1); image(woman_image); axis image; set(gca,'XTick',[],'YTick',[]);
title('Original')
cod_X_comp = wcodemat(X_comp,number_of_columns);
subplot(1,2,2); image(cod_X_comp); axis image; set(gca,'XTick',[],'YTick',[]);
title('Compressed using variable hard thresholds')
xlabel(sprintf('Energy retained = %2.1f%% \nNull coefficients = %2.1f%%',perfL2,perf0))

% Return to default settings.
dwtmode('zpd')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Polyphase filter implementation

% Determine the filters.
[low_pass_filter_decomposition,high_pass_filter_decomposition,low_pass_filter_reconstruction,high_pass_filter_reconstruction] ...
                                                                                                                = orthfilt(dbwavf('db2'));
decomposition_and_reconstruction_filter_length = length(low_pass_filter_decomposition);

% Load a signal.
load noisdopp
noisy_doppler_signal = noisdopp;
noisy_doppler_signal_length = length(noisy_doppler_signal);

% Change the signal into polyphase form.
xeven = dyaddown(noisy_doppler_signal,1);  % Even part
xodd = dyaddown(noisy_doppler_signal,0);   % Odd part 
xodd = [0 , xodd(1:noisy_doppler_signal_length/2-1)];
polyphased_doppler_signal = [xeven; xodd];

% Construct the polyphase matrix.
polyphase_decomposition_filter = zeros(2,2, decomposition_and_reconstruction_filter_length/2);
polyphase_decomposition_filter(1,1,:) = dyaddown(low_pass_filter_decomposition,1);  % h0,even[n]
polyphase_decomposition_filter(1,2,:) = dyaddown(low_pass_filter_decomposition,0);  % h0,odd[n]
polyphase_decomposition_filter(2,1,:) = dyaddown(high_pass_filter_decomposition,1);  % h1,even[n]
polyphase_decomposition_filter(2,2,:) = dyaddown(high_pass_filter_decomposition,0);  % h1,odd[n]

polyphase_decomposition_filter(:,:,1)
polyphase_decomposition_filter(:,:,2)

% Run the polyphase filter:
% Y = polyfilt(low_pass_interpolation_filter_fft,woman_image);
% % Polyphase filter implementation (2 channels)
% % 
% %   X = input signal, separated into even and odd phases.
% %       first row = even phase
% %	second row = odd phase	
% %   Y = output signal, separated into even and odd phases.
% %   H = 2x2 polyphase matrix
% %       H(1,1,:) = h0,even[n]
% %       H(1,2,:) = h0,odd[n]
% %       H(2,1,:) = h1,even[n]
% %       H(2,2,:) = h1,odd[n]
y0 = conv(squeeze(polyphase_decomposition_filter(1,1,:)),polyphased_doppler_signal(1,:)) + ...
    conv(squeeze(polyphase_decomposition_filter(1,2,:)),polyphased_doppler_signal(2,:));
y1 = conv(squeeze(polyphase_decomposition_filter(2,1,:)),polyphased_doppler_signal(1,:)) + ...
    conv(squeeze(polyphase_decomposition_filter(2,2,:)),polyphased_doppler_signal(2,:));
Y = [y0; y1];


% Plot the results.
levels_vec = noisy_doppler_signal_length/2;
n = 0:levels_vec-1;
clf
subplot(2,1,1)
plot(n,Y(1,1:levels_vec))
axis tight
xlabel('Sample number')
ylabel('Lowpass')
title('Output from polyphase filter')
subplot(2,1,2)
plot(n,Y(2,1:levels_vec))
axis tight
xlabel('Sample number')
ylabel('Highpass')


% Compute the results using the direct approach.
low_pass_level1_coefficients = dyaddown( conv(noisy_doppler_signal,low_pass_filter_decomposition) , 1);
high_pass_level1_coefficients = dyaddown( conv(noisy_doppler_signal,high_pass_filter_decomposition) , 1);

% Now compare the results.
clf
subplot(2,1,1)
plot(n,Y(1,1:levels_vec)-low_pass_level1_coefficients(1:levels_vec))
axis tight
xlabel('Sample number')
ylabel('Lowpass difference')
title('Difference in outputs produced by polyphase and direct forms')
subplot(2,1,2)
plot(n,Y(2,1:levels_vec)-high_pass_level1_coefficients(1:levels_vec))
axis tight
xlabel('Sample number')
ylabel('Highpass difference')


% Plot the determinant of the polyphase matrix as a function of frequency.
R = 32;
f_vec = 2/R*(-R/2:R/2-1);
H0even = fftshift(fft(polyphase_decomposition_filter(1,1,:),R));
H0odd = fftshift(fft(polyphase_decomposition_filter(1,2,:),R));
H1even = fftshift(fft(polyphase_decomposition_filter(2,1,:),R));
H1odd = fftshift(fft(polyphase_decomposition_filter(2,2,:),R));
delta = zeros(1,R);
delta(:) = H0even .* H1odd - H0odd .* H1even;
clf
plot(f_vec,abs(delta),'x-')
axis([-1 1 0 1.5])
xlabel('Angular frequency (normalized by pi)')
ylabel('Magnitude of determinant') %a constant value of 1 for all values
title('Determinant of the polyphase matrix')


% Verify that the filter is orthogonal i.e. Hp'(w*) Hp(w) = I
A11 = zeros(1,R);
A11(:) = abs(H0even).^2 + abs(H1even).^2;
A12 = zeros(1,R);
A12(:) = conj(H0even).*H0odd + conj(H1even).*H1odd;
A21 = zeros(1,R);
A21(:) = conj(H0odd).*H0even + conj(H1odd).*H1even;
A22 = zeros(1,R);
A22(:) = abs(H0odd).^2 + abs(H1odd).^2;
clf
subplot(4,1,1)
plot(f_vec,A11,'x-');
axis([-1 1 0 1.5])
ylabel('A11')
title('Variation of A = Hp''(w*) Hp(w) with frequency')
subplot(4,1,2)
plot(f_vec,A12,'x-')
axis([-1 1 0 1.5])
ylabel('A12')
subplot(4,1,3)
plot(f_vec,A21,'x-')
axis([-1 1 0 1.5])
ylabel('A21')
subplot(4,1,4)
plot(f_vec,A22,'x-')
axis([-1 1 0 1.5])
xlabel('Angular frequency (normalized by pi)')
ylabel('A22')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Example 6: Compute the samples of Daubechies scaling function and
% wavelet using the inverse DWT.

p = 2;                                     % Number of zeros at pi.
scaling_function_support = 2 * p - 1;                            % Support of the scaling function
number_of_levels = 5;                            % Number of iterations/levels.
discretization_parameter = 2^number_of_levels;
number_of_valid_samples = discretization_parameter * scaling_function_support;
low_pass_filter_reconstruction = daub(scaling_function_support+1) / 2; % Synthesis lowpass filter - the wavelet is the scaling filter???
high_pass_filter_reconstruction = (-1).^[0:scaling_function_support]' .* flipud(low_pass_filter_reconstruction);% Synthesis highpass filter.

% For the scaling function, we need to compute the inverse DWT with a delta
% for the approximation coefficients.  (All detail coefficients are set to zero.)
y = upcoef('a',[1;0],low_pass_filter_reconstruction,high_pass_filter_reconstruction,number_of_levels);    % Inverse DWT.
phi_scaling = discretization_parameter * [0; y(1:number_of_valid_samples)]; %the scaling starts from zero, 
plot(y);                                                                 %beyond y(levels_vec) all zeros

% For the wavelet, we need to compute the inverse DWT with a delta for the
% detail coefficients.  (All approximation coefficients and all detail
% coefficients at finer scales are set to zero.)
y = upcoef('d',[1;0],low_pass_filter_reconstruction,high_pass_filter_reconstruction,number_of_levels);    % Inverse DWT.
psi_wavelet = discretization_parameter * [0; y(1:number_of_valid_samples)];
plot(y);

% Determine the time vector.
t_vec = [0:number_of_valid_samples]' / discretization_parameter;

% Plot the results.
plot(t_vec,phi_scaling,'-',t_vec,psi_wavelet,'--')
legend('Scaling function','Wavelet')
title('Scaling function and wavelet by iteration of synthesis filter bank.')
xlabel('t')


% Now compute the scaling function and wavelet by recursion.
% phivals (not part of the Matlab toolbox) does this. 
%CHANGE THIS FUNCTION TO NAME:   get_scaling_and_wavelet_function_from_filter_coefficients!!!
[t1,phi_scaling2,psi_wavelet2] = phivals(daub(2*p),number_of_levels);

% Plot the results.
plot(t1,phi_scaling2,'-',t1,psi_wavelet2,'--')
legend('Scaling function','Wavelet')
title('Scaling function and wavelet by recursion.')
xlabel('t')


% View the scaling functions side by side.
plot(t_vec,phi_scaling,'-',t1,phi_scaling2,'--')
legend('Scaling function using iteration','Scaling function using recursion')
title('Comparison of the two methods (recursion is exact.)')
xlabel('t')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


????????????????????????????????
% Example 3a: Compute the samples of the biorthogonal scaling functions
% and wavelets.

[reconstruction_filter_scaling,decomposition_filter_scaling] = biorwavf('bior2.2');           % 9/7 filters
[low_pass_filter_decomposition,high_pass_filter_decomposition,low_pass_filter_reconstruction,high_pass_filter_reconstruction] ...
                                               = biorfilt(decomposition_filter_scaling, reconstruction_filter_scaling);

[x_vec,phi_scaling,phi_scaling_tilde,psi_wavelet,psi_wavelet_tilde] = ...
    biphivals(low_pass_filter_decomposition,high_pass_filter_decomposition,low_pass_filter_reconstruction,high_pass_filter_reconstruction,5);

plot(x_vec,phi_scaling,'-',x_vec,psi_wavelet,'-.')
legend('Primary scaling function', 'Primary wavelet')


plot(x_vec,phi_scaling_tilde,'--',x_vec,psi_wavelet_tilde,':')
legend('Dual scaling function', 'Dual wavelet')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Approximation of a functions by the scaling function and its translates. 

p = 1;
wavelet_function = daub(2*p);
%h = [-1 0 9 16 9 0 -1 0]'/16;
scaling_function_support = length(wavelet_function);
[x1,phi_scaling] = phivals(wavelet_function,4);

number_of_valid_samples = 2^4;
number_of_valid_samples = 12;
n = number_of_valid_samples * number_of_valid_samples;
x_vec = (0:n)/n*6;
sf = [phi_scaling; zeros((number_of_valid_samples-scaling_function_support+1)*2^4,1)];

for num = 1:5

  y = zeros(size(sf));
  tmp = 1/n * ((-scaling_function_support+2)*number_of_valid_samples : ...
              (-scaling_function_support+2+32)*number_of_valid_samples-1)';
  v = x_vec'/6;
  if num == 1
    ck = [zeros(scaling_function_support-2,1); wavelet_function; zeros(number_of_valid_samples,1)];
  elseif num == 2
    f_function = ones(n+1,1);
  elseif num == 3
    f_function = v;
    ck = scalecoeffs(tmp,32,wavelet_function,0,4);
  elseif num == 4
    f_function = 4*v.*v-4*v+1;
    ck = scalecoeffs(4*tmp.*tmp-4*tmp+1,32,wavelet_function,0,4);
  elseif num == 5
    f_function = -6*v.*v.*v+9*v.*v-3*v;
    ck = scalecoeffs(-6*tmp.*tmp.*tmp+9*tmp.*tmp-3*tmp,32,wavelet_function,0,4);
  end

  clf
  minval = 0;
  maxval = 0;
  for k = -scaling_function_support+2:number_of_valid_samples-1
    if num == 2
      g = eoshift(sf,k*number_of_valid_samples);
    else
      g = ck(k+scaling_function_support-1)*eoshift(sf,k*number_of_valid_samples);
    end
    hold on
    plot(x_vec,g,':')
    hold off
    y = y + g;
    minval = min(minval,min(g));   
    maxval = max(maxval,max(g));    
  end
  hold on
  plot(x_vec,y)
  hold off

  minval = min(0,min(y));
  maxval = max(y);
  reconstruction_filter_scaling = maxval - minval;
  minval = minval - 0.2 * reconstruction_filter_scaling;
  maxval = maxval + 0.2 * reconstruction_filter_scaling;
  axis([min(x_vec) max(x_vec) minval maxval])
  xlabel('x')
  ylabel('f(x)')
  if num == 1
    title('Representation of a scaling function by its translates')
    v = axis;
    v(2) = 6;
    axis(v);
  elseif num == 2
    title('Representation of a constant function by translates of a scaling function')
  elseif num == 3
    title('Representation of a linear function by translates of a scaling function')
  elseif num == 4
    title('Representation of a quadratic by translates of a scaling function')
  elseif num == 5
    title('Representation of a cubic by translates of a scaling function')
  end

  if num > 1
    
    plot(x_vec(1:n),f_function(1:n)-y(1:n))
    xlabel('x')
    ylabel('f(x)-f_{approx}(x)')
    title('Approximation error')
  end
  

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Example 4: Examine how polynomial data behaves in the filter bank
% when the lowpass filter has p zeros at pi.

p = 3;              % number of zeros at pi.
%a = [1 1]/128;
polynomial_coefficients = [1 3 3 1]/128;    % Coefficients of the polynomial (low order to high order.)
x_vec = 0:127;
polynomial_degree = length(polynomial_coefficients) - 1;  % Degree of the polynomial.
signal_length = length(x_vec);
polynomial_signal = zeros(1,signal_length);
current_axis_raised_to_certain_power = ones(1,signal_length);
for k = 1:polynomial_degree+1
  polynomial_signal = polynomial_signal + polynomial_coefficients(k) * current_axis_raised_to_certain_power;
  current_axis_raised_to_certain_power = current_axis_raised_to_certain_power .* x_vec;
end

% Compute the DWT.
scaling_function_support = 2 * p;
low_pass_filter_decomposition = daub(scaling_function_support);
high_pass_filter_decomposition = (-1).^[0:scaling_function_support-1]' .* flipud(low_pass_filter_decomposition);
[low_pass_level1_coefficients,high_pass_level1_coefficients] = dwt(polynomial_signal,low_pass_filter_decomposition,high_pass_filter_decomposition);

% Plot the results.
clf
subplot(3,1,1);
plot(polynomial_signal)
axis([0 signal_length-1 min(polynomial_signal) max(polynomial_signal)])
title(sprintf('Wavelet transform of degree %d polynomial data.  H0(w) has %d zeros at pi.', polynomial_degree, p))
ylabel('x')
subplot(3,1,2);
plot(low_pass_level1_coefficients)
axis([0 ,signal_length-1 ,min(low_pass_level1_coefficients) ,max(low_pass_level1_coefficients)])
ylabel('y0')
subplot(3,1,3);
plot(high_pass_level1_coefficients)
minval = min(0,min(high_pass_level1_coefficients(p:length(high_pass_level1_coefficients)-p+1)));
maxval = max(0,max(high_pass_level1_coefficients(p:length(high_pass_level1_coefficients)-p+1)));
reconstruction_filter_scaling = maxval - minval;
minval = minval - 0.2*reconstruction_filter_scaling;
maxval = maxval + 0.2*reconstruction_filter_scaling;
axis([0 signal_length-1 minval maxval])
ylabel('y1')

fprintf('Maximum value of y1 (excluding boundary effects) = %d.\n',max(abs(high_pass_level1_coefficients(p:length(high_pass_level1_coefficients)-p+1))))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Treatment of boundaries.

clf

fd = fopen('barbara.raw', 'r');
polyphased_doppler_signal = fread(fd, [512,512], 'uchar');
polyphased_doppler_signal = polyphased_doppler_signal';
map = (0:255)'/255 * ones(1,3);
colormap(map)
imagesc(uint8(polyphased_doppler_signal))
axis image; set(gca,'XTick',[],'YTick',[]); title('Single stage decomposition')
title('Original image')


% We will use the 9/7 filters with symmetric extension at the
% boundaries.
dwtmode('zpd')  % zpd, per, sym, sp0, sp1
%wname = 'db5'
psi_wavelet = 'bior4.4';

% Compute a 2-level decomposition of the image using the 9/7 filters.
[wavelet_coefficients_mat,reconstruction_filter_scaling] = wavedec2(polyphased_doppler_signal,2,psi_wavelet);

% Extract the level 1 coefficients.
approximation_coefficients_level1 = appcoef2(wavelet_coefficients_mat,reconstruction_filter_scaling,psi_wavelet,1);         
high_pass_filter_decomposition = detcoef2('h',wavelet_coefficients_mat,reconstruction_filter_scaling,1);           
detail_coefficients_vertical_level1 = detcoef2('v',wavelet_coefficients_mat,reconstruction_filter_scaling,1);           
detail_coefficients_diagonal_level1 = detcoef2('d',wavelet_coefficients_mat,reconstruction_filter_scaling,1);           

% Extract the level 2 coefficients.
approximation_coefficients_level2 = appcoef2(wavelet_coefficients_mat,reconstruction_filter_scaling,psi_wavelet,2);
detail_coefficients_horizontal_level2 = detcoef2('h',wavelet_coefficients_mat,reconstruction_filter_scaling,2);
detail_coefficients_vertical_level2 = detcoef2('v',wavelet_coefficients_mat,reconstruction_filter_scaling,2);
detail_coefficients_diagonal_level2 = detcoef2('d',wavelet_coefficients_mat,reconstruction_filter_scaling,2);

% Display the decomposition up to level 1 only.
image(uint8([approximation_coefficients_level1/2,high_pass_filter_decomposition*10;detail_coefficients_vertical_level1*10,detail_coefficients_diagonal_level1*10]))
axis image; set(gca,'XTick',[],'YTick',[]);
title('Single stage decomposition')


% Display the entire decomposition upto level 2.
st = dwtmode('status','nodisp');
if strcmp(st,'per')
  image(uint8([[approximation_coefficients_level2/4,detail_coefficients_horizontal_level2*10;detail_coefficients_vertical_level2*10,detail_coefficients_diagonal_level2*10],high_pass_filter_decomposition*10;detail_coefficients_vertical_level1*10,detail_coefficients_diagonal_level1*10]));
  axis image; set(gca,'XTick',[],'YTick',[]);
  title('Two stage decomposition')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

















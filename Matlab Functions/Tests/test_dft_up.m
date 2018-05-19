function [col_shift,row_shift,CCmax] = test_dft_up()

N=512/4;
view_size = N;
x0=0;
y0=0;
sigma_x=N/2;
sigma_y=N/2;
rho=0;
background=15;
amplitude=5;
ratio = 0.01;
upsampling = 5;
accuracy = 100;
spacing = view_size/N;
x_original = view_size*(1/N)*[-ceil((N-1)/2) :1: floor((N-1)/2)];
[X_original,Y_original] = meshgrid(x_original);
gaussian_2D_centered = ...
   background + amplitude*exp(-(X_original-x0).^2/(2*sigma_x^2)-(Y_original-y0).^2/(2*sigma_y^2));
x_zoomed = 1+(view_size/N)*(1/N*2)*[-ceil((N-1)/2):1:floor((N-1)/2)];
[X_zoomed,Y_zoomed] = meshgrid(x_zoomed);
N_initial_upsample = ceil(accuracy*1.5);
x_zoomed_upsampled = 1+(view_size/N*2)*(1/N_initial_upsample)*[-ceil((N_initial_upsample-1)/2):1:floor((N_initial_upsample-1)/2)];
[X_zoomed_upsampled,Y_zoomed_upsampled] = meshgrid(x_zoomed_upsampled);
gaussian_2D_centered_splined_zoomed = interp2(X_original,Y_original,gaussian_2D_centered,X_zoomed,Y_zoomed,'spline');
% gaussian_2D_centered_zoomed = real(interp2_ft_over_grid(gaussian_2D_centered,X_original,Y_original,X_zoomed,Y_zoomed));
% gaussian_2D_centered_zoomed_upsampled = real(interp2_ft_over_grid(gaussian_2D_centered,X_original,Y_original,X_zoomed_upsampled,Y_zoomed_upsampled));
gaussian_2D_centered_zoomed_upsampled = real(interp2_ft_grid(gaussian_2D_centered,X_original,Y_original,X_zoomed_upsampled,Y_zoomed_upsampled));
gaussian_2D_centered_splined_zoomed_upsampled = interp2(X_original,Y_original,gaussian_2D_centered,X_zoomed_upsampled,Y_zoomed_upsampled,'spline');
gaussian_2D_centered_zoomed_analytic = ...
   background + amplitude*exp(-(X_zoomed-x0).^2/(2*sigma_x^2)-(Y_zoomed-y0).^2/(2*sigma_y^2));
gaussian_2D_centered_zoomed_upsampled_analytic = ...
   background + amplitude*exp(-(X_zoomed_upsampled-x0).^2/(2*sigma_x^2)-(Y_zoomed_upsampled-y0).^2/(2*sigma_y^2));
figure;
subplot(2,2,1)
imagesc(gaussian_2D_centered); 
title('original gaussian');
colorbar;
subplot(2,2,2)
imagesc(gaussian_2D_centered_splined_zoomed);
title('original zoomed using spline');  
colorbar;
subplot(2,2,3)
imagesc(gaussian_2D_centered_zoomed_upsampled);
title('original upsampled using interp2-ft-over-grid');
colorbar;
subplot(2,2,4)
imagesc(gaussian_2D_centered_zoomed_analytic);
colorbar;
title('analytic zoomed');

figure;
subplot(3,1,1)
imagesc(gaussian_2D_centered_splined_zoomed);
title('original zoomed using spline');
colorbar;
subplot(3,1,2)
imagesc(gaussian_2D_centered_zoomed_analytic);
title('analytic zoomed');
colorbar;
subplot(3,1,3)
imagesc(gaussian_2D_centered_zoomed_analytic-gaussian_2D_centered_splined_zoomed);
title('difference analytic splined');
colorbar;

figure;
subplot(3,1,1) 
imagesc(gaussian_2D_centered_zoomed_upsampled);
title('original zoomed and upsampled using spline');
colorbar;
subplot(3,1,2)
imagesc(gaussian_2D_centered_zoomed_upsampled_analytic);
title('analytic zoomed and upsampled');
colorbar;
subplot(3,1,3)
imagesc(gaussian_2D_centered_zoomed_upsampled_analytic-gaussian_2D_centered_zoomed_upsampled);
title('difference analytic splined');
colorbar; 
 
% First upsample by a factor of 2 to obtain initial estimate 
% Embed Fourier data in a 2x larger array
buf1ft=fft2(gaussian_2D_centered);
[m,n]=size(buf1ft);
mlarge=m*2;   
nlarge=n*2;  
CC=zeros(mlarge,nlarge);
CC(m+1-fix(m/2):m+1+fix((m-1)/2),n+1-fix(n/2):n+1+fix((n-1)/2)) = ft2(gaussian_2D_centered,1);
 
% Compute crosscorrelation and locate the peak
%     CC = abs(ifft2(ifftshift(CC))); % Calculate cross-correlation
CC = abs(ift2(CC,1/mlarge))*2^2;
[max1,loc1] = max(CC); 
[max2,loc2] = max(max1);
rloc=loc1(loc2);cloc=loc2;
CCmax=CC(rloc,cloc);
% figure;
% imagesc(CC);
% title('gaussian upsampled by 2 using zero padding');
% colorbar;

% Obtain shift in original pixel grid from the position of the
% cross-correlation peak
[m,n] = size(CC); md2 = fix(m/2); nd2 = fix(n/2);
if rloc > md2
    row_shift = rloc - m - 1;
else
    row_shift = rloc - 1;
end
if cloc > nd2
    col_shift = cloc - n - 1;
else
    col_shift = cloc - 1;
end
row_shift=row_shift/2; %we devide by two because in the beginning we upsampled the matrices by 2!
col_shift=col_shift/2;
 
%%% DFT computation %%%
% Initial shift estimate in upsampled grid 
row_shift = round(row_shift*accuracy)/accuracy;
col_shift = round(col_shift*accuracy)/accuracy;
dftshift = fix(ceil(accuracy*1.5)/2); %% Center of output array at dftshift+1
% row_shift=0;
% col_shift=0;
% Matrix multiply DFT around the current shift estimate
CC = conj(dftups(buf1ft,ceil(accuracy*1.5),ceil(accuracy*1.5),accuracy,...
    dftshift-row_shift*accuracy,dftshift-col_shift*accuracy))/(md2*nd2);
% CC = conj(dftups(buf1ft,N,N,N,...
%     50-row_shift*N,50-col_shift*N))/(md2*nd2);
figure;  
subplot(2,2,1)
imagesc(change_values_range(real(CC),0,1));
title('original upsampled using dft-up and real');
colorbar;
subplot(2,2,2)
imagesc(change_values_range(gaussian_2D_centered_splined_zoomed_upsampled,0,1));
title('original splined and upsampled');
colorbar;
subplot(2,2,3)
% imagesc(change_values_range(real(CC),0,1)-change_values_range(gaussian_2D_centered_splined_zoomed_upsampled,0,1));
imagesc(change_values_range(gaussian_2D_centered_splined_zoomed_upsampled,0,1)-change_values_range(gaussian_2D_centered_zoomed_upsampled_analytic,0,1));
title('splined - analytic');
colorbar;
subplot(2,2,4)
imagesc(change_values_range(real(CC),0,1)-change_values_range(gaussian_2D_centered_zoomed_upsampled_analytic,0,1));
title('DFT - analytic');
colorbar;

figure;
subplot(3,1,1)
imagesc(real(CC));
title('original upsampled using dft-up and real');
colorbar;
subplot(3,1,2)
imagesc(gaussian_2D_centered_zoomed_upsampled);
title('original upsampled using interp2-ft-over-grid');
colorbar;
subplot(3,1,3)
imagesc(gaussian_2D_centered_zoomed_upsampled-real(CC));
title('difference between interp2-ft-over-grid and dft-up');
colorbar;

% Locate maximum and map back to original pixel grid
[max1,loc1] = max(CC);
[max2,loc2] = max(max1);
rloc = loc1(loc2); cloc = loc2;
CCmax = CC(rloc,cloc);

rg00 = dftups(buf1ft.*conj(buf1ft),1,1,1,0,0)/(md2*nd2);
% rf00 = dftups(buf2ft.*conj(buf2ft),1,1,1,0,0)/(md2*nd2);

CCmax = abs(CCmax)/(rg00);

rloc = rloc - dftshift - 1;
cloc = cloc - dftshift - 1;
row_shift = row_shift + rloc/accuracy;
col_shift = col_shift + cloc/accuracy;

%i saw that this returns the negative of the real translation:
row_shift=-row_shift;
col_shift=-col_shift;

%the given translations are from the zero to the center of the image
%(unlike what happens when i use fft2 and ifft2 to calculate cross correlation):
row_shift = row_shift - center_row;
col_shift = col_shift - center_col;

row_shift=row_shift*spacing;
col_shift=col_shift*spacing;
1;

close('all');
%         CCmax = interp2(X,Y,cross_corr,col_shift,row_shift)/sqrt(norm1*norm2);


function out=dftups(in,nor,noc,accuracy,roff,coff)
[nr,nc]=size(in);
% Compute kernels and obtain DFT by matrix products
kernc=exp((-2*pi*1i/(nc*accuracy))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
kernr=exp((-2*pi*1i/(nr*accuracy))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
out=kernr*in*kernc;
return






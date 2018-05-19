function [interpolated_fft_mat] = fft_large_matrix_small_frequencies_oriented_fft(input_matrix, padding_factor, upsample_factor, flag_centered, col_section, row_section)

% N = 512;
% final_pseudo_padded_N = 4096/2;
% padding_factor = final_pseudo_padded_N/N;
% upsample_factor = 2;
% input_matrix = make_gaussian_beams_profile(20,0,0,N,1).*exp(1i*10*randn(N,N));
% col_section = 2;
% row_section = 1;
% flag_centered = 0;

final_pseudo_padded_N = N*padding_factor;

% %ORIGINAL FFT NO PADDING:
% tic
% bloblo = ft2(input_matrix,1);
% toc
% figure;
% imagesc(abs(bloblo));
% title('original fft no padding');


% %FFT WITH PADDING:
% tic
% bla = pad_array_to_certain_size(input_matrix,final_pseudo_padded_N*upsample_factor);
% blabla = ft2(bla,1);
% toc
% figure
% imagesc(abs(blabla));
% title('padded fft');


% %CREATE MANY REPLICAS OF THE ABOVE UN-PADDED FFT:
% tic
% dftshift_row = fix(final_pseudo_padded_N/2);
% dftshift_col = fix(final_pseudo_padded_N/2);
% [nr,nc]=size(input_matrix);
% kernc=exp((-2*pi*1i/(nr))*( ifftshift([0:1:(nc-1)*1]).' - floor(nc/2*1) )*( [0:upsample_factor:final_pseudo_padded_N-1] - dftshift_col )/upsample_factor);
% kernr=exp((-2*pi*1i/(nr))*( [0:upsample_factor:final_pseudo_padded_N-1].' - dftshift_row )/upsample_factor*( ifftshift([0:1:(nr-1)*1]) - floor(nr/2*1)  ));
% interpolated_fft_mat2=kernr*input_matrix*kernc;
% toc
% figure
% imagesc(abs(interpolated_fft_mat2));
% title('created many replicates of the above upsampled and padded fft with each replica not upsampled');
% 1;


%GET UPSAMPLED CENTER PART OF THE FIRST UNPADDED FFT:
tic
N=N*upsample_factor;
if flag_centered==1
    dftshift_row = fix(N/2);
    dftshift_col = fix(N/2);
else
    dftshift_row = (N/2)*padding_factor - N*(col_section-1);
    dftshift_col = (N/2)*padding_factor - N*(row_section-1);
end
%TRY TO GET UNREPLICATED PADDED FFT WITH RESULAR SAMPLING:
[nr,nc]=size(input_matrix);
kernc=exp((-2*pi*1i/(nr*padding_factor*upsample_factor))*( ifftshift([0:1:(nc-1)*1]).' - floor(nc/2*1) )*( [0:1:N-1] - dftshift_col )/1);
kernr=exp((-2*pi*1i/(nr*padding_factor*upsample_factor))*( [0:1:N-1].' - dftshift_row )/1*( ifftshift([0:1:(nr-1)*1]) - floor(nr/2*1)  ));
interpolated_fft_mat2=kernr*fftshift(input_matrix)*kernc;
% toc
% figure
% imagesc(abs(interpolated_fft_mat2));
% title('created an regular sampled copy of the middle of the upsampled padded fft');
% 1;


% %CHECK WHETHER I CAN USE DFT TO CALCULATE THE 2D FFT RESULTING FROM AN
% %ARBITRARILY LOCATED SMALL SECTION OF THE 2D ORIGINAL IMAGE!?!??
% function [out]=dftups(in,final_N,roff,coff)
% %Computes the IDFT over a given grid of input DFT-ed matrix (in).
% [nr,nc]=size(in);
% % Compute kernels and obtain DFT by matrix products
% kernc=exp((-2*pi*1i/(final_N))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:final_N-1] - coff ));
% kernr=exp((-2*pi*1i/(final_N))*( [0:final_N-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
% out=kernr*in*kernc;
% return



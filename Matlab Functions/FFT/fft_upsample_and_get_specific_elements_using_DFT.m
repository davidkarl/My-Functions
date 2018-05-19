function [interpolated_fft_mat] = fft_upsample_and_get_specific_elements_using_DFT(input_matrix, final_pseudo_padded_N, row_start,row_stop,col_start,col_stop)

% N_original = 512;
% final_pseudo_padded_N = 4096/2;
% upsample_factor = final_pseudo_padded_N/N_original;
% input_matrix = make_gaussian_beams_profile(20,0,0,N_original,1).*exp(1i*10*randn(N_original,N_original));
% row_start=1;
% row_stop=700;
% col_start=1;
% col_stop=700;

% final_pseudo_padded_N = N_original*upsample_factor;

% %ORIGINAL FFT NO PADDING:
% tic
% bloblo = ft2(input_matrix,1);
% toc
% figure;
% imagesc(abs(bloblo));
% title('original fft no padding');


% %FFT WITH PADDING:
% tic
% bla = pad_array_to_certain_size(input_matrix,final_pseudo_padded_N);
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
% tic
new_N_col = col_stop - col_start + 1;
new_N_row = row_stop - row_start + 1;

dftshift_row = fix(final_pseudo_padded_N/2) - (row_start-1);
dftshift_col = fix(final_pseudo_padded_N/2) - (col_start-1);

%TRY TO GET UNREPLICATED PADDED FFT WITH RESULAR SAMPLING:
[nr,nc]=size(input_matrix);
kernc=exp((-2*pi*1i/(final_pseudo_padded_N))*( ifftshift([0:1:(nc-1)*1]).' - floor(nc/2*1) )*( [0:1:new_N_col-1] - dftshift_col )/1);
kernr=exp((-2*pi*1i/(final_pseudo_padded_N))*( [0:1:new_N_row-1].' - dftshift_row )/1*( ifftshift([0:1:(nr-1)*1]) - floor(nr/2*1)  ));
interpolated_fft_mat=kernr*fftshift(input_matrix)*kernc;
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



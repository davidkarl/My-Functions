% function [zoomed_in_matrix] = zoom_in_using_fast_fourier_interpolation(input_matrix,zoom_in_factor)
%THIS COULD BE DON'E MORE EFFICIENTLY BUT FUCK IT.

%shift matrix to allow zooming in to center and not corner:
% input_matrix = shift_matrix(input_matrix,1,-size(input_matrix,2)/4,-size(input_matrix,1)/4);

% %zoom in:
% existing_rows = size(input_matrix,1);
% existing_cols = size(input_matrix,2);
% wanted_number_of_rows = size(input_matrix,1);
% wanted_number_of_cols = size(input_matrix,2);
% needed_shift = -zoom_in_factor;
% kernel_c=exp((-1i*2*pi/(nc*accuracy))*( ifftshift([0:existing_cols-1]).' - floor(existing_cols/2) )*( [0:wanted_number_of_cols-1] - needed_shift ));
% kernel_r=exp((-1i*2*pi/(nr*accuracy))*( [0:wanted_number_of_rows-1].' - needed_shift )*( ifftshift([0:existing_rows-1]) - floor(existing_rows/2)  ));
% 
% zoom_in_matrix = kernel_r.*fft2(fliplr(flipud(input_matrix))).*kernel_c;

%check:
clear all;
clc;
N = 512;
zoom_in_factor = 2;
size_factor = 1.4;
input_matrix = create_speckles_of_certain_size_in_pixels(90,N,1,0);
input_matrix2 = shift_matrix(input_matrix,1,size(input_matrix,2)*(-1/2+1/zoom_in_factor/2),size(input_matrix,1)*(-1/2+1/zoom_in_factor/2));
% input_matrix2 = shift_matrix(input_matrix,1,size(input_matrix,2)*(-1/2+1/zoom_in_factor/size_factor/2),size(input_matrix,1)*(-1/2+1/zoom_in_factor/size_factor/2));
% input_matrix2 = input_matrix;
existing_rows = size(input_matrix,1);
existing_cols = size(input_matrix,2);
wanted_number_of_rows = round(size(input_matrix,1)*size_factor);
wanted_number_of_cols = round(size(input_matrix,2)*size_factor);
figure(1)
imagesc(abs(input_matrix));
title('original');
% figure(2)
% imagesc(abs(input_matrix2));
% title('shifted');
% input_matrix = interpft(input_matrix,zoom_in_factor*N,1);
% input_matrix = interpft(input_matrix,zoom_in_factor*N,2);
% N=N*zoom_in_factor;
% input_matrix = input_matrix(N/2-N/zoom_in_factor/2+1:N/2+N/2/zoom_in_factor,N/2-N/zoom_in_factor/2+1:N/2+N/zoom_in_factor/2);
% figure(2) 
% imagesc(abs(input_matrix)) 
% title('artificially zoomed in');
%   
   
needed_shift = -zoom_in_factor; 
% needed_shift = fix(ceil(zoom_in_factor*N)/2);
kernel_c=exp((-1i*2*pi/(wanted_number_of_rows*zoom_in_factor))*( ifftshift([0:existing_cols-1]).' - floor(existing_cols/2) )*( [0:wanted_number_of_cols-1] - needed_shift ));
kernel_r=exp((-1i*2*pi/(wanted_number_of_rows*zoom_in_factor))*( [0:wanted_number_of_rows-1].' - needed_shift )*( ifftshift([0:existing_rows-1]) - floor(existing_rows/2)  ));
zoomed_in_matrix = kernel_r*fft2(fliplr(flipud(input_matrix2)))*kernel_c;

%check:
N_original = size(input_matrix,1);
% input_matrix = interpft(input_matrix,zoom_in_factor*N,1);
% input_matrix = interpft(input_matrix,zoom_in_factor*N,2);
% N=N*zoom_in_factor;
% input_matrix = input_matrix(N/2-N/zoom_in_factor/2:N/2+N/2/zoom_in_factor-1,N/2-N/zoom_in_factor/2:N/2+N/zoom_in_factor/2-1);
% N = N_original;
% input_matrix = interpft(input_matrix,round(size_factor*N),1);
% input_matrix = interpft(input_matrix,round(size_factor*N),2);
template = linspace(-N/2,N/2,N);
[X,Y]=meshgrid(template);
new_template = linspace(-N/zoom_in_factor/2,(N/2)/zoom_in_factor,round(N*size_factor));
[X_new,Y_new] = meshgrid(new_template);
input_matrix = interp2(X,Y,input_matrix,X_new,Y_new);
  
input_matrix = input_matrix/max(max(abs(input_matrix)));
zoomed_in_matrix = zoomed_in_matrix/max(max(abs(zoomed_in_matrix)));
figure(2)
imagesc(abs(input_matrix));
colorbar;
figure(3) 
imagesc(abs(zoomed_in_matrix));
colorbar;
% [col_shift,row_shift,CCmax] = return_shifts_with_fourier_sampling(abs(zoomed_in_matrix),abs(input_matrix),1,1000);
figure(4)
difference_mat = abs(zoomed_in_matrix)-abs(input_matrix);
imagesc(difference_mat);
colorbar;
1;


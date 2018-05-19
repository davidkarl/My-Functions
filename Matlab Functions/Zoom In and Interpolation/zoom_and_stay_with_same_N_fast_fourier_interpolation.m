% function [zoomed_in_matrix] = zoom_and_stay_with_same_N_fast_fourier_interpolation(input_matrix,zoom_in_factor)
%THIS COULD BE DON'E MORE EFFICIENTLY BUT FUCK IT.

% %input example:
% N = 512;
% zoom_in_factor = 1.7;
% input_matrix = create_speckles_of_certain_size_in_pixels(50,N,1,0);

% shift matrix to allow zooming in to center and not corner:
input_matrix = shift_matrix(input_matrix,1,size(input_matrix,2)*(-1/2+1/zoom_in_factor/2),size(input_matrix,1)*(-1/2+1/zoom_in_factor/2));

%zoom in:
existing_rows = size(input_matrix,1);
existing_cols = size(input_matrix,2);
wanted_number_of_rows = size(input_matrix,1);
wanted_number_of_cols = size(input_matrix,2);
needed_shift = -zoom_in_factor;
kernel_c=exp((-1i*2*pi/(nc*accuracy))*( ifftshift([0:existing_cols-1]).' - floor(existing_cols/2) )*( [0:wanted_number_of_cols-1] - needed_shift ));
kernel_r=exp((-1i*2*pi/(nr*accuracy))*( [0:wanted_number_of_rows-1].' - needed_shift )*( ifftshift([0:existing_rows-1]) - floor(existing_rows/2)  ));

zoom_in_matrix = kernel_r.*fft2(fliplr(flipud(input_matrix))).*kernel_c;


% function [zoomed_in_matrix] = interp2_ft_center_including_zoom_out(input_matrix,zoom_in_factor,size_factor)
%THIS COULD BE DON'E MORE EFFICIENTLY BUT FUCK IT.

%DEFINITIONS:
%zoom_in_factor = how much to zoom into image
%size_factor = N_final/N_initial
 
%INPUT EXAMPLE:
clear all;
clc;
N = 512;
zoom_in_factor = 0.63;
size_factor=2.2;
input_matrix = create_speckles_of_certain_size_in_pixels(50,N,1,0);
N_rows = size(input_matrix,1);
N_cols = size(input_matrix,2);
original_input_matrix = input_matrix;



% shift matrix to allow zooming in to center and not corner:
input_matrix = shift_matrix(input_matrix,1,size(input_matrix,2)*(-1/2+1/zoom_in_factor/2),size(input_matrix,1)*(-1/2+1/zoom_in_factor/2));

%zoom in:
existing_rows = size(input_matrix,1);
existing_cols = size(input_matrix,2);
wanted_number_of_rows = round(size(input_matrix,1)*size_factor);
wanted_number_of_cols = round(size(input_matrix,2)*size_factor);
fractional_addition_rows = wanted_number_of_rows/(1+zoom_in_factor/(1-zoom_in_factor))/2;
fractional_addition_cols = wanted_number_of_cols/(1+zoom_in_factor/(1-zoom_in_factor))/2;

%update zoom in factor if it's zoom out:
needed_shift = -zoom_in_factor-1+1/size_factor;
% needed_shift = fix(ceil(zoom_in_factor*N)/2);
kernel_c=exp((-1i*2*pi/(wanted_number_of_rows*zoom_in_factor))*( ifftshift([0:existing_cols-1]).' - floor(existing_cols/2) )*( [0:wanted_number_of_cols-1] - needed_shift ));
kernel_r=exp((-1i*2*pi/(wanted_number_of_rows*zoom_in_factor))*( [0:wanted_number_of_rows-1].' - needed_shift )*( ifftshift([0:existing_rows-1]) - floor(existing_rows/2)  ));
zoomed_in_matrix = kernel_r*fft2(fliplr(flipud(input_matrix)))*kernel_c;
 
%cut off excess pixels if zoom out:
if zoom_in_factor<1
   N = wanted_number_of_rows;
   zoomed_in_matrix(1:fix(fractional_addition_rows),1:end) = 0;
   zoomed_in_matrix(1:end,1:fix(fractional_addition_cols)) = 0;
   zoomed_in_matrix(end-fix(fractional_addition_rows):end,1:end) = 0;
   zoomed_in_matrix(1:end,end-fix(fractional_addition_cols):end) = 0;
end


%check:
template = linspace(-N/2,N/2,N_rows);
[X,Y]=meshgrid(template);
new_template = linspace(-(N/2)/zoom_in_factor,(N/2)/zoom_in_factor,wanted_number_of_rows);
[X_new,Y_new] = meshgrid(new_template);
interpolated_matrix = interp2(X,Y,original_input_matrix,X_new,Y_new,'cubic spline');
original_input_matrix = original_input_matrix/max(max(abs(original_input_matrix)));
interpolated_matrix = interpolated_matrix/max(max(abs(interpolated_matrix)));
zoomed_in_matrix = zoomed_in_matrix/max(max(abs(zoomed_in_matrix)));
figure(1)
imagesc(abs(original_input_matrix));
colorbar;  
figure(2) 
imagesc(abs(interpolated_matrix));
colorbar; 
figure(3) 
imagesc(abs(zoomed_in_matrix));
colorbar; 
figure(4)
difference_mat = abs(zoomed_in_matrix)-abs(interpolated_matrix);
imagesc(difference_mat); 
a=max(abs(difference_mat(:)));
colorbar;       
1;                                    
     

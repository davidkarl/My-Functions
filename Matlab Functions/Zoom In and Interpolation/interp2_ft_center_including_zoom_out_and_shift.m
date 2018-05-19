function [zoomed_in_matrix] = interp2_ft_center_including_zoom_out_and_shift(input_matrix,zoom_in_factor,size_factor)
%THIS COULD BE DONE MORE EFFICIENTLY BUT FUCK IT.

%DEFINITIONS:
%zoom_in_factor = how much to zoom into image
%size_factor = N_final/N_initial
 
%INPUT EXAMPLE:
clear all;
clc;
N = 512;
zoom_in_factor = 2;
size_factor=1;
ROI_shift_x = 53.23; %[pixels]
ROI_shift_y = 0;
speckle_size = 10;
input_matrix = create_speckles_of_certain_size_in_pixels(speckle_size,N,1,0);
N_rows = size(input_matrix,1);
N_cols = size(input_matrix,2);
original_input_matrix = input_matrix;



% %shift matrix to take account of original_image_shift:
input_matrix = shift_matrix(input_matrix,1,-ROI_shift_x,ROI_shift_y);

%shift matrix to allow zooming in to center and not corner:
input_matrix = shift_matrix(input_matrix,1,size(input_matrix,2)*(-1/2+1/zoom_in_factor/2),size(input_matrix,1)*(-1/2+1/zoom_in_factor/2));



%zoom in:
existing_rows = size(input_matrix,1);
existing_cols = size(input_matrix,2);
wanted_number_of_rows = round(size(input_matrix,1)*size_factor);
wanted_number_of_cols = round(size(input_matrix,2)*size_factor);

% fractional_addition_rows = wanted_number_of_rows/( 1+(zoom_in_factor/(1-zoom_in_factor)) )/2;
% fractional_addition_cols = wanted_number_of_cols/( 1+(zoom_in_factor/(1-zoom_in_factor)) )/2;
% upper_ROI_row = 1 + fractional_addition_rows - original_image_shift_y*size_factor;
% lower_ROI_row = (N_rows-original_image_shift_y)*size_factor - fractional_addition_rows;
% left_ROI_col = 1 + fractional_addition_cols + original_image_shift_x*size_factor;
% right_ROI_col = (N_cols-original_image_shift_x)*size_factor - fractional_addition_rows;

%if zoom_in_factor<1 (zoom out)--> fractional_addition>0 because i add fractional cells
fractional_addition_rows = N_rows*( 1/zoom_in_factor - 1 )/2;
fractional_addition_cols = N_cols*( 1/zoom_in_factor - 1 )/2;

%find image cut-off places:
%the variables represent the indices of the original image within the 
%new view size scaled to N_originals:
upper_ROI_row = (1 + ROI_shift_y + fractional_addition_rows);
lower_ROI_row = (N_rows + ROI_shift_y - fractional_addition_rows);
left_ROI_col = (1 - ROI_shift_x + fractional_addition_cols);
right_ROI_col = (N_cols - ROI_shift_x - fractional_addition_rows);

%build original meshgrid:
x_original = linspace(-N/2,N/2,N);
[X_original,Y_original] = meshgrid(x_original);
Y_original = -1*Y_original; %make cartesian
spacing_original = x_original(2)-x_original(1);
%build view ROI meshgrid:
x_view = linspace(-N/2/zoom_in_factor,N/2/zoom_in_factor,wanted_number_of_rows);
spacing_view = x_view(2)-x_view(1);
[X_view,Y_view] = meshgrid(x_view);
Y_view = -1*Y_view; %make cartesian
X_view = X_view + ROI_shift_x;
Y_view = Y_view + ROI_shift_y;

%find right_col_position:
right_col_position = 1+(X_original(1,end)-X_view(1,1))/spacing_view;
%find left_col_position:
left_col_position = 1+(X_original(1,1)-X_view(1,1))/spacing_view;
%find upper_row_position:
upper_row_position = 1+(Y_view(1,1)-Y_original(1,1))/spacing_view;
%find lower_row_position:
lower_row_position = 1+(Y_view(1,1)-Y_original(end,1))/spacing_view;
  
% %scale:
% right_col_position = right_col_position*size_factor;
% left_col_position = left_col_position*size_factor;
% upper_row_position = upper_row_position*size_factor;
% lower_row_position = lower_row_position*size_factor;

% %scale:
% upper_ROI_row = upper_ROI_row*size_factor;
% lower_ROI_row = lower_ROI_row*size_factor;
% left_ROI_col = left_ROI_col*size_factor;
% right_ROI_col = right_ROI_col*size_factor;

%update zoom in factor if it's zoom out:
needed_shift = -zoom_in_factor-1+1/size_factor;
% needed_shift = fix(ceil(zoom_in_factor*N)/2);
kernel_c=exp((-1i*2*pi/(wanted_number_of_rows*zoom_in_factor))*( ifftshift([0:existing_cols-1]).' - floor(existing_cols/2) )*( [0:wanted_number_of_cols-1] - needed_shift ));
kernel_r=exp((-1i*2*pi/(wanted_number_of_rows*zoom_in_factor))*( [0:wanted_number_of_rows-1].' - needed_shift )*( ifftshift([0:existing_rows-1]) - floor(existing_rows/2)  ));
zoomed_in_matrix = kernel_r*fft2(fliplr(flipud(input_matrix)))*kernel_c;
 
%MAKE MATRIX ZERO IF VIEW IS OUTSIDE ORIGINAL MATRIX:
if round(left_col_position)>wanted_number_of_cols || round(upper_row_position)>wanted_number_of_rows || ...
        round(lower_row_position)<1 || round(right_col_position)<1
    zoomed_in_matrix(1:end,1:end) = 0;
end 
% figure(1)
% imagesc(abs(original_input_matrix));
% figure(2) 
% imagesc(abs(zoomed_in_matrix));

%FILL WITH ZEROS THE PROPER CELLS:
if round(left_col_position)>1
   zoomed_in_matrix(1:end,1:round(left_col_position)) = 0; 
end
if round(upper_row_position)>1
   zoomed_in_matrix(1:round(upper_row_position),1:end) = 0;
end
if round(lower_row_position)<wanted_number_of_rows
   zoomed_in_matrix(round(lower_row_position):end,1:end) = 0;
end
if round(right_col_position)<wanted_number_of_cols
   zoomed_in_matrix(1:end,round(right_col_position):end) = 0; 
end
% if round(left_col_position)>1
%    zoomed_in_matrix(1:end,1:round(left_col_position)-1) = 0; 
% end
% if round(upper_row_position)>1
%    zoomed_in_matrix(1:round(upper_row_position)-1,1:end) = 0;
% end
% if round(lower_row_position)<wanted_number_of_rows
%    zoomed_in_matrix(round(lower_row_position)+1:end,1:end) = 0;
% end
% if round(right_col_position)<wanted_number_of_cols
%    zoomed_in_matrix(1:end,round(right_col_position)+1:end) = 0; 
% end


zoomed_in_matrix = rot90(zoomed_in_matrix,3);


% figure(1)
% imagesc(abs(original_input_matrix));
% figure(3) 
% imagesc(abs(zoomed_in_matrix));
  
  

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
% a=max(abs(difference_mat(:)));
colorbar;       
1;                                    
     

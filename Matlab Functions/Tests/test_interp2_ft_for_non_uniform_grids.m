%Test interp2_ft for non uniform grids
%This is derived directly from the script interp2_ft


%DEFINITIONS:
%zoom_in_factor = how much to zoom into image
%size_factor = N_final/N_initial
%ROI_shift_x = how much to shift the center of the new grid in x (cartesian)
%ROI_shift_y = how much to shift the center of the new grid in y (cartesian)
%spacing = new grid spacing, this is JUST to easily state the ROI shifts.
%--> new_grid_spacing = old_grid_spacing/zoom_in_factor/size_factor

% INPUT EXAMPLE, un-comment for example:


%Define Image:
N = 100;
original_x = 1:N;
original_y = 1:N;
[original_X,original_Y] = meshgrid(original_x,original_y);
speckle_size = 10;
input_matrix = real(create_speckles_of_certain_size_in_pixels(speckle_size,N,1,0));
N_rows = size(input_matrix,1);
N_cols = size(input_matrix,2);
original_input_matrix = input_matrix;


%Zoom/Shift Parameters (old version of interp2_ft):
zoom_in_factor = 1;
size_factor=1;
spacing = 1;
ROI_shift_x = 1.2; %[pixels] 
ROI_shift_y = -0.12;

%New Parameters:
% wanted_x = 1:1.5:N;
% wanted_y = 1:1.5:N;
% [wanted_X,wanted_Y] = meshgrid(wanted_x,wanted_y);
shiftx = abs(create_speckles_of_certain_size_in_pixels(speckle_size,N,1,0));
shifty = abs(create_speckles_of_certain_size_in_pixels(speckle_size,N,1,0));
shiftx = shiftx/max(shiftx(:));
shifty = shifty/max(shifty(:));
sigma = 2;
wanted_X = original_X + sigma*shiftx;
wanted_Y = original_Y + sigma*shifty;
wanted_X_flattened = wanted_X(:);
wanted_Y_flattened = wanted_Y(:);


%definitions:
existing_rows = size(input_matrix,1);
existing_cols = size(input_matrix,2);
wanted_number_of_rows = round(size(input_matrix,1)*size_factor);
wanted_number_of_cols = round(size(input_matrix,2)*size_factor);

%shift matrix to allow zooming in to center and not corner and account for ROI center shift:
input_matrix = shift_matrix(input_matrix,1,size(input_matrix,2)*(-1/2+1/zoom_in_factor/2)-ROI_shift_x,size(input_matrix,1)*(-1/2+1/zoom_in_factor/2)+ROI_shift_y);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%RESAMPLE:
%NEED TO UNDERSTAND THIS, IT INTRODUCES A SMALL ERROR VS. INTERP2:
%(1). Old version:
needed_shift = -zoom_in_factor-0.5*zoom_in_factor*(size_factor-1); 
cols_pre_vector = ifftshift([0:existing_cols-1]).' - floor(existing_cols/2);
cols_post_vector = ([0:wanted_number_of_cols-1] - needed_shift)/(zoom_in_factor)  / (wanted_number_of_cols);
rows_pre_vector = ifftshift([0:existing_rows-1]) - floor(existing_rows/2);
rows_post_vector = ([0:wanted_number_of_rows-1].' - needed_shift)/(zoom_in_factor) / (wanted_number_of_rows);
%(2). New version: 
needed_shift = -zoom_in_factor-0.5*zoom_in_factor*(size_factor-1); 
cols_pre_vector = ifftshift([0:existing_cols-1]).' - floor(existing_cols/2);
cols_post_vector = (wanted_x)/(zoom_in_factor)  / (wanted_number_of_cols);
rows_pre_vector = ifftshift([0:existing_rows-1]) - floor(existing_rows/2);
rows_post_vector = (wanted_y.')/(zoom_in_factor) / (wanted_number_of_rows);
 
kernel_c=exp( (-1i*2*pi)*( cols_pre_vector )*( cols_post_vector ) );
kernel_r=exp( (-1i*2*pi)*( rows_post_vector )*( rows_pre_vector ) );
resampled_matrix = kernel_r*fft2(rot90(input_matrix,2))*kernel_c;
%(3). Head in the wall version for arbitrary shifts:
input_matrix_rotated_fft = fft2(rot90(input_matrix,2));
input_matrix_rotated_fft_flattened = input_matrix_rotated_fft(:);
kernel_c_flattened = exp( (-i1*2*pi)*

%NORMALIZE:
resampled_matrix = resampled_matrix/(size(resampled_matrix,1)*size(resampled_matrix,2));


interpolated_matrix = interp2(original_X,original_Y,original_input_matrix,wanted_X,wanted_Y);
figure(1);
imagesc(real(original_input_matrix));
title('original');
figure(2);
imagesc(real(resampled_matrix));
title('resampled');
figure(3);
imagesc(interpolated_matrix);
title('interpolated');



% %check: 
% template = linspace(-N/2,N/2-1,N_rows);
% [X,Y]=meshgrid(template);
% Y = -Y;
% new_template = linspace(-(N/2)/zoom_in_factor,(N/2-1)/zoom_in_factor,wanted_number_of_rows);
% new_spacing = new_template(2)-new_template(1);
% spacing_view = x_view(2)-x_view(1);
% [X_new,Y_new] = meshgrid(new_template); 
% Y_new = -Y_new; %cartesian
% X_new = X_new + ROI_shift_x*spacing; 
% Y_new = Y_new + ROI_shift_y*spacing;
% interpolated_matrix = interp2(X,Y,original_input_matrix,X_new,Y_new,'cubic spline');
% interpolated_matrix(isnan(interpolated_matrix))=0;
% original_input_matrix = original_input_matrix/max(max(abs(original_input_matrix)));
% interpolated_matrix = interpolated_matrix/max(max(abs(interpolated_matrix)));
% resampled_matrix = resampled_matrix/max(max(abs(resampled_matrix)));
% figure(1)
% imagesc(abs(original_input_matrix));
% title('original matrix');
% colorbar;  
% figure(2) 
% imagesc(abs(interpolated_matrix));
% title('interpolated matrix using interp2');
% colorbar; 
% figure(3) 
% imagesc(abs(resampled_matrix));
% title('resampled matrix using interp2 ft');
% colorbar; 
% figure(4)
% difference_mat = abs(resampled_matrix-interpolated_matrix);
% imagesc(difference_mat); 
% a=max(abs(difference_mat(:)));
% b=max(max(abs(difference_mat(2:end-1,2:end-1))));
% title('absolute difference mat');
% colorbar;


% [col_shift,row_shift,CCmax] = return_shifts_with_fourier_sampling(abs(resampled_matrix),abs(interpolated_matrix),1,1000);     
% 1;

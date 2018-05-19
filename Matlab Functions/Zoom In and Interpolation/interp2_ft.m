function [resampled_matrix] = interp2_ft(input_matrix,zoom_in_factor,ROI_shift_x,ROI_shift_y,spacing,size_factor)
%THIS COULD BE DONE MORE EFFICIENTLY BUT FUCK IT.
%THIS IS REALLY ONLY GOOD WHEN THERE ARE VERY HIGH FREQUENCIES AND A LARGE ZOOM

%DEFINITIONS:
%zoom_in_factor = how much to zoom into image
%size_factor = N_final/N_initial
%ROI_shift_x = how much to shift the center of the new grid in x (cartesian)
%ROI_shift_y = how much to shift the center of the new grid in y (cartesian)
%spacing = new grid spacing, this is JUST to easily state the ROI shifts.
%--> new_grid_spacing = old_grid_spacing/zoom_in_factor/size_factor

% INPUT EXAMPLE, un-comment for example:
N = 100;
zoom_in_factor = 2;
size_factor=1;
spacing = 1;
ROI_shift_x = 1.2; %[pixels] 
ROI_shift_y = -0.12;
speckle_size = 10;
input_matrix = create_speckles_of_certain_size_in_pixels(speckle_size,N,1,0);
N_rows = size(input_matrix,1);
N_cols = size(input_matrix,2);
original_input_matrix = input_matrix;



%translate shifts in grid spacing to shifts in pixels:
ROI_shift_x = ROI_shift_x/spacing;
ROI_shift_y = ROI_shift_y/spacing;

%definitions:
existing_rows = size(input_matrix,1);
existing_cols = size(input_matrix,2);
wanted_number_of_rows = round(size(input_matrix,1)*size_factor);
wanted_number_of_cols = round(size(input_matrix,2)*size_factor);

%shift matrix to allow zooming in to center and not corner and account for ROI center shift:
input_matrix = shift_matrix(input_matrix,1,size(input_matrix,2)*(-1/2+1/zoom_in_factor/2)-ROI_shift_x,size(input_matrix,1)*(-1/2+1/zoom_in_factor/2)+ROI_shift_y);


%build original meshgrid:
x_original = linspace(-existing_cols/2,(existing_cols/2-1),existing_cols);
y_original = linspace(-existing_rows/2,(existing_rows/2-1),existing_rows);
y_original = -1*y_original;
spacing_original = x_original(2)-x_original(1);
%build view ROI meshgrid:
x_view = linspace(-existing_cols/2/zoom_in_factor,(existing_cols/2-1)/zoom_in_factor,wanted_number_of_cols);
y_view = linspace(-existing_rows/2/zoom_in_factor,(existing_rows/2-1)/zoom_in_factor,wanted_number_of_rows);
y_view = -1*y_view;
spacing_view = x_view(2)-x_view(1);
x_view = x_view + ROI_shift_x;
y_view = y_view + ROI_shift_y;

%find right_col_position:
right_col_position = 1+(x_original(end)-x_view(1))/spacing_view;
%find left_col_position:
left_col_position = 1+(x_original(1)-x_view(1))/spacing_view;
%find upper_row_position:
upper_row_position = 1+(y_view(1)-y_original(1))/spacing_view;
%find lower_row_position:
lower_row_position = 1+(y_view(1)-y_original(end))/spacing_view;

%RESAMPLE:
%NEED TO UNDERSTAND THIS, IT INTRODUCES A SMALL ERROR VS. INTERP2:
needed_shift = -zoom_in_factor-0.5*zoom_in_factor*(size_factor-1); 
kernel_c=exp((-1i*2*pi/(wanted_number_of_cols*zoom_in_factor))*( ifftshift([0:existing_cols-1]).' - floor(existing_cols/2) )*( [0:wanted_number_of_cols-1] - needed_shift ));
kernel_r=exp((-1i*2*pi/(wanted_number_of_rows*zoom_in_factor))*( [0:wanted_number_of_rows-1].' - needed_shift )*( ifftshift([0:existing_rows-1]) - floor(existing_rows/2)  ));
resampled_matrix = kernel_r*fft2(rot90(input_matrix,2))*kernel_c;
 
%MAKE MATRIX ZERO IF VIEW IS OUTSIDE ORIGINAL MATRIX:
if round(left_col_position)>wanted_number_of_cols || round(upper_row_position)>wanted_number_of_rows || ...
        round(lower_row_position)<1 || round(right_col_position)<1
    resampled_matrix(1:end,1:end) = 0;
end 

%FILL WITH ZEROS THE PROPER CELLS:
if ceil(left_col_position)-1>1
   resampled_matrix(1:end,1:ceil(left_col_position)-1) = 0; 
end
if ceil(upper_row_position)-1>1
   resampled_matrix(1:ceil(upper_row_position)-1,1:end) = 0;
end
if floor(lower_row_position)+1<wanted_number_of_rows
   resampled_matrix(floor(lower_row_position)+1:end,1:end) = 0;
end
if floor(right_col_position)+1<wanted_number_of_cols
   resampled_matrix(1:end,floor(right_col_position)+1:end) = 0; 
end

%NORMALIZE:
resampled_matrix = resampled_matrix/(size(resampled_matrix,1)*size(resampled_matrix,2));

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

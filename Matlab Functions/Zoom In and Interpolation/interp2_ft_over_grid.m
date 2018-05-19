% function [resampled_matrix] = interp2_ft_over_grid(input_matrix,X_initial,Y_initial,X_final,Y_final)

%%%%%%
N=512;
input_matrix = abs(create_speckles_of_certain_size_in_pixels(50,N,1,0));
x_intial=linspace(-N/2,N/2,N);
[X_initial,Y_initial] = meshgrid(x_intial);
% x_final=linspace(-N+100,N-200,N);
x_final = linspace(-N/4,N/4,N);
[X_final,Y_final]=meshgrid(x_final);
%%%%%%

x_initial_start = X_initial(1,1);
x_initial_stop = X_initial(1,end);
y_initial_start = Y_initial(1,1);
y_initial_stop = Y_initial(end,1);
x_initial_middle = (x_initial_start+x_initial_stop)/2;
y_initial_middle = (y_initial_start+y_initial_stop)/2;
x_initial_size = x_initial_stop - x_initial_start;
y_initial_size = y_initial_stop - y_initial_start;
x_initial_spacing = X_initial(1,2)-X_initial(1,1);
y_initial_spacing = Y_initial(2,1)-Y_initial(1,1);

x_final_start = X_final(1,1);
x_final_stop = X_final(1,end);
y_final_start = Y_final(1,1);
y_final_stop = Y_final(end,1);
x_final_middle = (x_final_start+x_final_stop)/2;
y_final_middle = (y_final_start+y_final_stop)/2;
x_final_size = x_final_stop - x_final_start;
y_final_size = y_final_stop - y_final_start;
x_final_spacing = X_final(1,2)-X_final(1,1);
y_final_spacing = Y_final(2,1)-Y_final(1,1);


ROI_shift_x = (x_final_middle - x_initial_middle);
ROI_shift_y = -(y_final_middle - y_initial_middle); 
zoom_in_factor_x = x_initial_size / x_final_size;
zoom_in_factor_y = y_initial_size / y_final_size;
zoom_in_factor = zoom_in_factor_x; %for now, untill i correct interp2_ft
size_x_initial = size(X_initial,2);
size_x_final = size(X_final,2);
size_y_initial = size(Y_initial,1);
size_y_final = size(Y_final,1);
size_factor_x = size_x_final/size_x_initial;
size_factor_y = size_y_final/size_y_initial;
size_factor = size_factor_x; %for now, untill i correct interp2_ft
spacing = x_initial_spacing;

[resampled_matrix] = interp2_ft(input_matrix,zoom_in_factor,ROI_shift_x,ROI_shift_y,spacing,size_factor);



[resampled_matrix2] = interp2(X_initial,Y_initial,input_matrix,X_final,Y_final,'spline');
subplot(3,1,1)
imagesc(input_matrix);
colorbar;
subplot(3,1,2)
imagesc(real(resampled_matrix));
colorbar;
subplot(3,1,3)
imagesc(resampled_matrix2);
colorbar;
figure(2)
imagesc(abs(real(resampled_matrix)-resampled_matrix2));
colorbar;






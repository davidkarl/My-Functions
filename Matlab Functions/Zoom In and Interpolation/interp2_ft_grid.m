function [interpolated_mat] = interp2_ft_grid(input_matrix,X_initial,Y_initial,X_final,Y_final)


%%%%%%
N=512;
input_matrix = abs(create_speckles_of_certain_size_in_pixels(50,N,1,0));
x_intial=linspace(-N/2,N/2,N);
[X_initial,Y_initial] = meshgrid(x_intial);
% x_final=linspace(-N+100,N-200,N);
x_final = linspace(-N/4,N/4,N);
[X_final,Y_final]=meshgrid(x_final);
%%%%%%



%get initial and final grid data:
x_initial_start = X_initial(1,1);
x_initial_stop = X_initial(1,end);
y_initial_start = Y_initial(1,1); 
y_initial_stop = Y_initial(end,1);
x_initial_middle = (x_initial_start+x_initial_stop)/2;
y_initial_middle = (y_initial_start+y_initial_stop)/2;
x_final_start = X_final(1,1);
x_final_stop = X_final(1,end);
y_final_start = Y_final(1,1);
y_final_stop = Y_final(end,1);
x_final_middle = (x_final_start+x_final_stop)/2;
y_final_middle = (y_final_start+y_final_stop)/2;

%find shift between centers:
centers_shift_in_original_pixels_col = X_final(1,size(X_final,2)+1-fix(size(X_final,2)/2))- X_initial(1,size(X_initial,2)+1-fix(size(X_initial,2)/2));
centers_shift_in_original_pixels_row = Y_final(size(Y_final,1)+1-fix(size(Y_final,1)/2),1)- Y_initial(size(Y_initial,1)+1-fix(size(Y_initial,1)/2),1);

%get accuracies:
[nor1,noc1] = size(input_matrix); 
[nor2,noc2] = size(X_final);
final_spacing_in_original_pixels_terms_row = (Y_final(2,1)-Y_final(1,1))/(Y_initial(2,1)-Y_initial(1,1));
final_spacing_in_original_pixels_terms_col = (X_final(1,2)-X_final(1,1))/(X_initial(1,2)-X_initial(1,1));
accuracy_row = 1/final_spacing_in_original_pixels_terms_row; %how much do i divide a pixel along row
accuracy_col = 1/final_spacing_in_original_pixels_terms_col; %how much do i divide a pixel along col


%get final dftshifts:
[original_center_row,original_center_col] = return_matrix_center(input_matrix);
dftshift_row = fix(nor2/2) + (original_center_row-1 - centers_shift_in_original_pixels_row)*accuracy_row;
dftshift_col = fix(noc2/2) + (original_center_col-1 - centers_shift_in_original_pixels_col)*accuracy_col;

% row_shift = round(row_shift*accuracy)/accuracy;
% col_shift = round(col_shift*accuracy)/accuracy;
% CC = conj(dftups(buf1ft,nor,noc,accuracy,...
%     dftshift-row_shift*accuracy,dftshift-col_shift*accuracy))/(md2*nd2);
 

%Original version:
interpolated_mat = real(conj(dftups_accuracy_two_axes(fft2(input_matrix),nor2,noc2,accuracy_row,accuracy_col,...
    dftshift_row,dftshift_col))/(noc1*nor1));

% interpolated_mat = abs(conj(dftups_accuracy_two_axes(fft2(input_matrix),nor2,noc2,accuracy_row,accuracy_col,...
%     dftshift_row,dftshift_col))/(noc1*nor1));

interpolated_mat = rot90(interpolated_mat,2);


% 
[resampled_matrix2] = interp2(X_initial,Y_initial,input_matrix,X_final,Y_final,'cubic spline');
subplot(3,1,1)
imagesc(input_matrix);
colorbar;
subplot(3,1,2)
imagesc(real(interpolated_mat));
colorbar;
subplot(3,1,3)
imagesc(resampled_matrix2);
colorbar;
figure(2)
imagesc(abs(real(interpolated_mat)-resampled_matrix2));
colorbar;


 
% 
% function out=dftups(in,nor,noc,accuracy_row,accuracy_col,roff,coff)
% %Computes the IDFT over a given grid of input DFT-ed matrix (in).
% [nr,nc]=size(in);
% % Compute kernels and obtain DFT by matrix products
% kernc=exp((-2*pi*1i/(nc*accuracy_col))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
% kernr=exp((-2*pi*1i/(nr*accuracy_row))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
% out=kernr*in*kernc;
% return
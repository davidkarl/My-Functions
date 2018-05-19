function [fft_without_DC] = fft_cut_matrix_DC(mat,delta)

fft_without_DC = ft2(mat,delta);
filter_center = ceil(size(mat,1)/2) + (1-mod(size(mat,1),2));
fft_without_DC(filter_center,filter_center)=0;


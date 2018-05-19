function [pyramid_height] =  get_wavelet_pyramid_height(index_matrix)
% [HEIGHT] = wpyrHt(INDICES)
%
% Compute height of separable QMF/wavelet pyramid with given index matrix.

%index matrix contains all the bands numbered oridnarily, but in order to
%get the number of levels (height) from it we need to know the number of
%bands per level and calculate:
if ( index_matrix(1,1) == 1 || index_matrix(1,2) == 1)
	number_of_bands_per_level = 1;
else
	number_of_bands_per_level = 3;
end

pyramid_height = (size(index_matrix,1)-1)/number_of_bands_per_level;

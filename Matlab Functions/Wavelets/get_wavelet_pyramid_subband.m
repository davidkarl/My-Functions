function im =  get_wavelet_pyramid_subband(wavelet_pyramid,index_matrix,level,band_number_at_wanted_level)
% RES = wpyrBand(PYR, INDICES, LEVEL, BAND)
%
% Access a subband from a separable QMF/wavelet pyramid.
%
% LEVEL (optional, default=1) indicates the scale (finest = 1,
% coarsest = wpyrHt(INDICES)).
%
% BAND (optional, default=1) indicates which subband (1=horizontal,
% 2=vertical, 3=diagonal).


if ~exist('level','var')
    level = 1;
end
if ~exist('band','var')
    band_number_at_wanted_level = 1;
end

if ( index_matrix(1,1) == 1 || index_matrix(1,2) == 1)
    number_of_bands_in_each_level = 1; %1D
else
    number_of_bands_in_each_level = 3; %2D full
end

if ((band_number_at_wanted_level > number_of_bands_in_each_level) || (band_number_at_wanted_level < 1))
    error(sprintf('Bad band number (%d) should be in range [1,%d].', band_number_at_wanted_level, number_of_bands_in_each_level));
end

max_level = get_wavelet_pyramid_height(index_matrix);
if ((level > max_level) || (level < 1))
    error(sprintf('Bad level number (%d), should be in range [1,%d].', level, max_level));
end

%the get_pyramid_subband matrix is built to get the variable band as an
%ordinary variable from 1 to N (total number of bands):
band = band_number_at_wanted_level + number_of_bands_in_each_level*(level-1);
im = get_pyramid_subband(wavelet_pyramid,index_matrix,band);

function [mat_in_lowpassed , mat_in_highpassed] = riesz_prefilter(mat_in, riesz_transform_object)
% Pre filter the image before applying the isostropic wavelet pyramid
%
% --------------------------------------------------------------------------
% Input arguments:
%
% A image to process
%
% CONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform.
%
% --------------------------------------------------------------------------
% Output arguments:
% 
% PA low frequency image obtained by filtering A
% RESIDUAL high frequency image
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
% 
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if ~isempty(riesz_transform_object.prefilter.filterLow),
    mat_in_fft = fft2(mat_in);
    mat_in_lowpassed = ifft2(mat_in_fft .* riesz_transform_object.prefilter.filterLow);
    mat_in_highpassed = ifft2(mat_in_fft .* riesz_transform_object.prefilter.filterHigh);
else
    mat_in_lowpassed = mat_in;
    mat_in_highpassed = zeros(size(mat_in));
end;
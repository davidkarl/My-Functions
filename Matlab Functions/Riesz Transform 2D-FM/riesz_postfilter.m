function combined_image = riesz_postfilter(lowpass_image, highpass_residual, riesz_transform_object)
% RIESZPOSTFILTER combine a low frequency image with a high frequency image
%
% PA = RieszPostfilter(A, residual, config)
% combine a low frequency image A with a high frequency image.
% A is processed with the lowpass prefilter object in config, while
% residual is processed with the highpass counterpart.
%
% --------------------------------------------------------------------------
% Input arguments:
%
% A low frequency image
%
% RESIDUAL high frequency image 
%
% CONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform.
%
% --------------------------------------------------------------------------
% Output arguments: 
%
% PA image obtained by combining the A and residual after filtering.
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if riesz_transform_object.prefilterType == PrefilterType.None,
    combined_image = lowpass_image;
else
    combined_image = ifft2(fft2(lowpass_image).*riesz_transform_object.prefilter.filterLow + ...
                           fft2(highpass_residual).*riesz_transform_object.prefilter.filterHigh);
end
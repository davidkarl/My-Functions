function riesz_coefficients_matrix = riesz_transform_analysis(mat_in, riesz_transform_object)
% RIESZANALYSIS perform the Riesz tranform of high order
%
% --------------------------------------------------------------------------
% Input arguments:
%
% A image to analyze
%
% CONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform.
%
% --------------------------------------------------------------------------
% Output arguments:
%
% Q Riesz coefficients. It consits in a 3D matrix
% whose 3rd dimension corresponds to Riesz channels.
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

%% perform Riesz transform
if riesz_transform_object.riesz_transform_order > 0
 
    mat_in_fft = fft2(mat_in);
    riesz_coefficients_matrix = zeros(size(mat_in, 1), size(mat_in, 2), riesz_transform_object.number_of_riesz_channels);
    
    if riesz_transform_object.flag_real_data,
        for riesz_channel_counter = 1:riesz_transform_object.number_of_riesz_channels,
            riesz_coefficients_matrix(:,:,riesz_channel_counter) = ...
                      real(ifft2(mat_in_fft.*riesz_transform_object.riesz_transform_filters{riesz_channel_counter}));
        end 
    else
        for riesz_channel_counter = 1:riesz_transform_object.number_of_riesz_channels,
            riesz_coefficients_matrix(:,:,riesz_channel_counter) = ...
                          ifft2(mat_in_fft.*riesz_transform_object.riesz_transform_filters{riesz_channel_counter});
        end
    end

else
    riesz_coefficients_matrix = mat_in;
end
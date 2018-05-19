function riesz_coefficients_matrix = riesz_transform_synthesis(riesz_coefficients_matrix, riesz_transform_object)
% RIESZSYNTHESIS perform the backward 3D Riesz transform
%
% --------------------------------------------------------------------------
% Input arguments:
%
% C1 Riesz coefficients. It consits in a 3D matrix
% whose 3rd dimension corresponds to Riesz channels.
%
% CONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform.
%
% --------------------------------------------------------------------------
% Output arguments:
%
%  C1 reconstructed image
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
% 
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if riesz_transform_object.riesz_transform_order>0,
    C1temp = riesz_coefficients_matrix;
    riesz_coefficients_matrix = zeros(riesz_transform_object.mat_size);
    for riesz_channel_counter = 1:riesz_transform_object.number_of_riesz_channels,
        riesz_coefficients_current_fft = fftn(C1temp(:, :, riesz_channel_counter));
        riesz_coefficients_matrix = riesz_coefficients_matrix + ...
             ifftn(riesz_coefficients_current_fft.*conj(riesz_transform_object.riesz_transform_filters{riesz_channel_counter}));
    end;
    if riesz_transform_object.flag_real_data,
        riesz_coefficients_matrix = real(riesz_coefficients_matrix);
    end
end
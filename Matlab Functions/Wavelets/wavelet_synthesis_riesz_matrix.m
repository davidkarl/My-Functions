function C1 = wavelet_synthesis_riesz_matrix(riesz_wavelet_coefficients_cell_array, riesz_transform_object)
%WAVELETSYNTHESIS perform wavelet reconstruction of Riesz-wavelet coefficients
%
% --------------------------------------------------------------------------
% Input arguments:
%
% Q cell of Riesz-wavelet coefficients. Each element of the cell is a 3D
% matrix that corresponds of the Riesz coefficients at a given scale. The
% third dimension of each matrix corresponds to Riesz channels.
%
% CONFIG RieszConfig2D object that specifies the primary Riesz-wavelet
% transform. 
%
% --------------------------------------------------------------------------
% Output arguments:
%
% C1 matrix of Riesz coefficients. It consists in a 3D matrix whose 3rd
% dimension corresponds to Riesz channels.
%
% --------------------------------------------------------------------------
% 
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if riesz_transform_object.number_of_riesz_channels < 2, %no riesz analysis
    switch riesz_transform_object.wavelet_type,
        case WaveletType.isotropic,
            riesz_wavelet_highpass_coefficients_cell_array  = cell(1, riesz_transform_object.number_of_scales);
            for scale_counter = 1:riesz_transform_object.number_of_scales,
                riesz_wavelet_highpass_coefficients_cell_array{scale_counter} = riesz_wavelet_coefficients_cell_array{scale_counter};
            end
            C1 = isotropic_band_limited_synthesis(riesz_wavelet_coefficients_cell_array{riesz_transform_object.number_of_scales+1}, ...
                                                  riesz_wavelet_highpass_coefficients_cell_array, 2, riesz_transform_object.isotropic_wavelet_type, 1);
        case WaveletType.spline
            error('2D spline wavelets are not included in this toolbox, please consider using isotropic wavelets instead');
        otherwise,
            [~, s] = enumeration('WaveletType');
            str = 'unkwnown wavelet type. Valid options are: ';
            for riesz_channel_counter = 1:length(s)
                str = strcat([str, s{riesz_channel_counter}, ', ']);
            end
            error(str);
    end
else % process riesz-wavelet coefficients
    switch riesz_transform_object.wavelet_type,
        case WaveletType.isotropic,
            C1 = zeros(riesz_transform_object.mat_size(1), riesz_transform_object.mat_size(2), riesz_transform_object.number_of_riesz_channels);
            for riesz_channel_counter = 1:riesz_transform_object.number_of_riesz_channels,
                riesz_wavelet_highpass_coefficients_cell_array = cell(1, riesz_transform_object.number_of_scales);
                for scale_counter = 1:riesz_transform_object.number_of_scales,
                    riesz_wavelet_highpass_coefficients_cell_array{scale_counter} = riesz_wavelet_coefficients_cell_array{scale_counter}(:,:,riesz_channel_counter);
                end
                C1(:,:,riesz_channel_counter) = isotropic_band_limited_synthesis(...
                                                    riesz_wavelet_coefficients_cell_array{riesz_transform_object.number_of_scales+1}(:,:,riesz_channel_counter), ...
                                                                 riesz_wavelet_highpass_coefficients_cell_array, ...
                                                                 2, riesz_transform_object.isotropic_wavelet_type, 1);
            end
        case WaveletType.spline
            error('2D spline wavelets are not included in this toolbox, please consider using isotropic wavelets instead');
        otherwise,
            [~, s] = enumeration('WaveletType');
            str = 'unkwnown wavelet type. Valid options are: ';
            for riesz_channel_counter = 1:length(s)
                str = strcat([str, s{riesz_channel_counter}, ', ']);
            end
            error(str);
    end
end

if riesz_transform_object.flag_real_data,
    C1 = real(C1);
end
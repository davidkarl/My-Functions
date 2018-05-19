function riesz_wavelet_coefficients_cell_array = wavelet_analysis_riesz_matrix(mat_in_or_riesz_coefficients_matrix_3D, ...
                                                                               riesz_transform_object)
%WAVELETANALYSIS perform wavelet decomposition of Riesz coefficients
%
% --------------------------------------------------------------------------
% Input arguments:
%
% R matrix of Riesz coefficients. It consists in a 3D matrix whose 3rd
% dimension corresponds to Riesz channels.
%
% CONFIG RieszConfig2D object that specifies the primary Riesz-wavelet
% transform.
%
% --------------------------------------------------------------------------
% Output arguments:
%
% Q cell of Riesz-wavelet coefficients. Each element of the cell is a 3D
% matrix that corresponds of the Riesz coefficients at a given scale.
%
% --------------------------------------------------------------------------
% 
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

%Get riesz transform parameters:
riesz_transform_order = riesz_transform_object.riesz_transform_order;
number_of_scales = riesz_transform_object.number_of_scales;
isotropic_wavelet_type = riesz_transform_object.isotropic_wavelet_type;
wavelet_type = riesz_transform_object.wavelet_type;
number_of_riesz_channels = riesz_transform_object.number_of_riesz_channels;
mat_in_number_of_dimensions = 2;
flag_coefficients_in_fourier_0_or_spatial_domain_1 = 1;


if (number_of_scales==0)
    iter = 1;
    riesz_wavelet_coefficients_cell_array{iter}{1} = mat_in_or_riesz_coefficients_matrix_3D;
else
    if riesz_transform_order < 1 % no riesz analysis
        switch wavelet_type,
            case WaveletType.isotropic,
                [LP , riesz_wavelet_coefficients_cell_array] = ...
                            isotropic_band_limited_analysis(mat_in_or_riesz_coefficients_matrix_3D, ...
                                                            mat_in_number_of_dimensions, ...
                                                            number_of_scales, ...
                                                            isotropic_wavelet_type, ...
                                                            flag_coefficients_in_fourier_0_or_spatial_domain_1);
                riesz_wavelet_coefficients_cell_array{number_of_scales+1} = LP;
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
    else %process riesz coefficients
        switch wavelet_type,
            case WaveletType.isotropic,
                riesz_wavelet_coefficients_cell_array = cell(1, number_of_scales+1);
                for riesz_channel_counter = 1:number_of_riesz_channels,
                    [LP , HP] =  isotropic_band_limited_analysis(mat_in_or_riesz_coefficients_matrix_3D(:,:,riesz_channel_counter), ...
                                                                 mat_in_number_of_dimensions, ...
                                                                 number_of_scales, ...
                                                                 isotropic_wavelet_type, ...
                                                                 flag_coefficients_in_fourier_0_or_spatial_domain_1);
                                                             
                    for scale_counter = 1:number_of_scales,
                        riesz_wavelet_coefficients_cell_array{scale_counter}(:,:, riesz_channel_counter)  = HP{scale_counter};
                    end
                    riesz_wavelet_coefficients_cell_array{number_of_scales+1}(:,:,riesz_channel_counter) = LP;
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
end
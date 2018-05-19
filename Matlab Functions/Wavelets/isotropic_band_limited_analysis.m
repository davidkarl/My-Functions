function [lowpass_part , highpass_wavelet_coefficients] = ...
                            isotropic_band_limited_analysis(mat_in, ...
                                                            mat_number_of_dimensions, ...
                                                            number_of_levels, ...
                                                            waveletType, ...
                                                            flag_input_output_coefficients_in_fourier_0_or_spatial_domain_1)
%ISOTROPICBANDLIMITEDANALYSIS perform multidimensional isotropic wavelet decomposition
%
% --------------------------------------------------------------------------
% Input arguments:
%
% IM multidimensional image to decompose
% 
% DIM dimension of the image. Should be 1, 2 or 3.
%
% J number of wavelet scales
%
% WAVELETTYPE element of the IsotropicWaveletType enumeration. Specifies
% the radial wavelet function.
% Optional. Default is IsotropicWaveletType.getDefaultValue
%
% SPATIALDOAMAIN 1 if image and coefficients are stored in the spatial domain,
% or in the Fourier domain (0). Optional. Default is 1.
%
% --------------------------------------------------------------------------
% Output arguments:
%
% LP lowpass filter image.
%
% HP wavelet coefficients. It consists in a cell of matrices, each of
% which is a wavelet band.
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if (mat_number_of_dimensions>3)
    error('Wavelet transform for 1d, 2d and 3d data only');
end
 
switch mat_number_of_dimensions,
    case 2,
        if (size(mat_in,1)~=size(mat_in, 2))
            error('Isotropic image required for wavelet analysis');
        end
    case 3,
        if (size(mat_in,1)~=size(mat_in, 2) || size(mat_in,1)~=size(mat_in, 3))
            error('Isotropic volume required for wavelet analysis');
        end
end

if (mod(size(mat_in, 1), 2^number_of_levels)~=0)
    error('The size of the image has to be a multiple of 2^J');
end

if nargin <4
    waveletType = IsotropicWaveletType.getDefaultValue;
end
if nargin <5
    flag_input_output_coefficients_in_fourier_0_or_spatial_domain_1 = 1;
end

if flag_input_output_coefficients_in_fourier_0_or_spatial_domain_1==1
    mat_in = fftn(mat_in);
end

highpass_wavelet_coefficients = cell(1, number_of_levels);
for level_counter = 1:number_of_levels
    
    %compute mask:
    switch waveletType,
        case IsotropicWaveletType.Meyer,
            [maskHP maskLP] =  meyerMask(size(mat_in, 1), size(mat_in,2), size(mat_in,3));
        case IsotropicWaveletType.Simoncelli,
            [maskHP maskLP] =  simoncelliMask(size(mat_in, 1), size(mat_in,2), size(mat_in,3));
        case IsotropicWaveletType.Papadakis,
            [maskHP maskLP] =  papadakisMask(size(mat_in, 1), size(mat_in,2), size(mat_in,3));
        case IsotropicWaveletType.Aldroubi,
            [maskHP maskLP] =  aldroubiMask(size(mat_in, 1), size(mat_in,2), size(mat_in,3));
        case IsotropicWaveletType.Shannon,
            [maskHP maskLP] =  halfSizeEllipsoidalMask(size(mat_in, 1), size(mat_in,2), size(mat_in,3));
        case IsotropicWaveletType.Ward,
			error('Ward s wavelet function is not provided in this toolbox');
        otherwise
            error('unknown wavelet type. Valid options are: meyer, simoncelli, papadakis, aldroubi, shannon')
    end
    
    %high pass image:
    fftHP = mat_in.*maskHP;
    %low pass image:
    mat_in = mat_in.*maskLP;
    
    %downSampling in spatial domain = fold the frequency domain!!!!!
    %multiple dimensions: cascade of downsampling in each cartesian direction
    switch mat_number_of_dimensions,
        case 1,
            c1 = size(mat_in,2)/2;
            mat_in = 0.5*(mat_in(1:c1) + mat_in((1:c1)+c1));
        case 2,
            c2 = size(mat_in,2)/2;
            c1 = size(mat_in,1)/2;
            mat_in = 0.25*(mat_in(1:c1, 1:c2) + mat_in((1:c1)+c1, 1:c2) + ...
                           mat_in((1:c1) + c1, (1:c2) +c2) + mat_in(1:c1, (1:c2) +c2));
        otherwise,
            c3 = size(mat_in,3)/2;
            c2 = size(mat_in,2)/2;
            c1 = size(mat_in,1)/2;
            temp = zeros(c1,c2,c3);
            for i= 0:1
                for j=0:1
                    for k= 0:1
                        temp = temp + mat_in((1:c1) + i*c1, (1:c2) + j*c2, (1:c3) + k*c3);
                    end
                end
            end
            mat_in = temp/8;
    end
    
    if flag_input_output_coefficients_in_fourier_0_or_spatial_domain_1==1
        highpass_wavelet_coefficients{level_counter} = ifftn(fftHP);
    else
        highpass_wavelet_coefficients{level_counter} = fftHP;
    end
end %LEVELS/SCALES LOOP

if flag_input_output_coefficients_in_fourier_0_or_spatial_domain_1==1
    lowpass_part = ifftn(mat_in);
else
    lowpass_part = mat_in;
end
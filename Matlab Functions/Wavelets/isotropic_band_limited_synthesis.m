function lowpass_residual = isotropic_band_limited_synthesis(lowpass_residual, highpass_wavelet_coefficients, ...
                                                             number_of_mat_dimensions, waveletType, flag_fourier_0_or_spatial_domain_1)
%ISOTROPICBANDLIMITEDSYNTHESIS perform multidimensional isotropic wavelet
%reconstruction
%
% --------------------------------------------------------------------------
% Input arguments:
%
% LP lowpass filter image.
%
% HP wavelet coefficients. It consists in a cell of matrices, each of
% which is a wavelet band.
%
% DIM dimension of the image. Should be 1, 2 or 3.
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
% LP multidimensional image that is reconstructed
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if number_of_mat_dimensions>3,
    error('Wavelet transform for 1d, 2d and 3d data only');
end

if nargin <4
    waveletType = IsotropicWaveletType.getDefaultValue;
end
if nargin <5
    flag_fourier_0_or_spatial_domain_1 = 1;
end

total_number_of_bands = size(highpass_wavelet_coefficients(:));

if flag_fourier_0_or_spatial_domain_1==1
    %MAT GIVEN IN THE SPATIAL DOMAIN:

    lowpass_band_fft = fftn(lowpass_residual);
    for band_counter = total_number_of_bands:-1:1
        %get highpass image:
        highpass_band_fft = fftn(highpass_wavelet_coefficients{band_counter});
        %upsample in the frequency domain:
        switch number_of_mat_dimensions,
            case 3,
                rep = [2 2 2];
            case 2,
                rep = [2 2];
            case 1,
                rep = [1 2];
        end
        lowpass_band_fft = repmat(lowpass_band_fft, rep);

        %compute filter:
        switch waveletType,
            case IsotropicWaveletType.Meyer,
                [maskHP maskLP] =  meyerMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Simoncelli,
                [maskHP maskLP] = simoncelliMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Papadakis,
                [maskHP maskLP] = papadakisMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Aldroubi,
                [maskHP maskLP] = aldroubiMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Ward,
				error('Ward s wavelet function is not provided in this toolbox');
            case IsotropicWaveletType.Shannon,
                [maskHP maskLP] =  halfSizeEllipsoidalMask(size(lowpass_band_fft, 1), size(lowpass_band_fft, 2), size(lowpass_band_fft,3));
            otherwise
                error('unknown wavelet type. Valid options are: meyer, simoncelli, papadakis, aldroubi, shannon')
        end
        %filter highpass and lowpass images:
        lowpass_band_fft = (2^number_of_mat_dimensions)*lowpass_band_fft.*maskLP + highpass_band_fft.*maskHP;
    end
    lowpass_residual = ifftn(lowpass_band_fft);

else
    %MAT GIVEN IN THE FOURIER DOMAIN:

    for band_counter = total_number_of_bands:-1:1
        %upsample in the frequency domain:
        switch number_of_mat_dimensions,
            case 3,
                rep = [2 2 2];
            case 2,
                rep = [2 2];
            case 1,
                rep = [1 2];
        end
        lowpass_band_fft = repmat(lowpass_residual, rep);
        %compute filter:
        switch waveletType,
            case IsotropicWaveletType.Meyer,
                [maskHP maskLP] =  meyerMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Papadakis,
                [maskHP maskLP] =  papadakisMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Aldroubi,
                [maskHP maskLP] =  aldroubiMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Simoncelli,
                [maskHP maskLP] = simoncelliMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Ward,
                [maskHP maskLP] = wardMask(size(lowpass_band_fft, 1), size(lowpass_band_fft,2), size(lowpass_band_fft,3));
            case IsotropicWaveletType.Shannon,
                [maskHP maskLP] =  halfSizeEllipsoidalMask(size(lowpass_band_fft, 1), size(lowpass_band_fft, 2), size(lowpass_band_fft,3));
            otherwise
                error('unknown wavelet type. Valid options are: meyer, simoncelli, papadakis, aldroubi, shannon')
        end 
        %filter highpass and lowpass images
        lowpass_residual = (2^number_of_mat_dimensions)*lowpass_band_fft.*maskLP + highpass_wavelet_coefficients{band_counter}.*maskHP;
    end
end



% RIESZCONFIG class of objects characterizing 2D Riesz-wavelet transforms
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

classdef riesz_transform_object,
    properties
        flag_real_data; % real valued data
        mat_size; % size of the volume 
        
        wavelet_type; % type of wavelet transform. See the WaveletType Enumeration
        isotropic_wavelet_type; % type of wavelet transform to use
        bspline_wavelet_transform_order; % order of the bspline wavelet transform (positive integer)
        number_of_scales; % number of scales for the wavelet transform
        
        riesz_transform_order; % order of the Riesz transform
        number_of_riesz_channels; % number of Riesz channels
        riesz_transform_filters; % filters for computing the Riesz transform
        %RieszOrders; % orders of the Riesz transform for each channel
        
        prefilterType; % prefiltering operation, see PrefilterType enumeration
        prefilter;
    end
    
    methods 

        function config = riesz_transform_object(dims, riesz_transform_order, number_of_scales, flag_prepare_filters)
            %Constructor:
            config.flag_real_data = 1;
            config.wavelet_type = WaveletType.getDefaultValue();
            config.isotropic_wavelet_type = IsotropicWaveletType.getDefaultValue();
            config.bspline_wavelet_transform_order = 1;
            config.prefilterType = PrefilterType.getDefaultValue();
            if nargin>0
                if (dims(1)~=dims(2) || dims(1)<1)
                    error('The Riesz-wavelet transform works only for isotropic volumes for now')
                end
                config.mat_size = dims;
            else
                config.mat_size = zeros(1,1,1);
            end
            if nargin>1,
                if (riesz_transform_order <0 || (riesz_transform_order-floor(riesz_transform_order)~=0))
                    error('The Riesz transform needs to be a positive integer');
                end
                config.riesz_transform_order=riesz_transform_order;
            else
                config.riesz_transform_order=1;
            end
            if nargin>2,
                if (number_of_scales <1 || (number_of_scales-floor(number_of_scales)~=0))
                    error('The number of scales needs to be a positive integer');
                end
                if (2^number_of_scales >= config.mat_size(1))
                    error('The number of scales is too large as compared to the image size');
                end
                config.number_of_scales = number_of_scales;
            else
                % set the number of scales such that the low-pass image is
                % at least of size 16x16
                config.number_of_scales = max(floor(log(config.mat_size(1)/16)/log(2)), 1);
            end
            if nargin>3 && flag_prepare_filters,
                config = prepareTransform(config);
            end
        end
        

        
        %Utility function:
        function config = prepareTransform(config, dims)
            if nargin>1,
                if (dims(1)~=dims(2) || dims(1)<1)
                    error('The Riesz-wavelet transform works only for non-empty isotropic volumes for now');
                end
                config.mat_size = dims;
            else
                if config.mat_size(1)<1,
                    error('Specify the volume size before initializing the filters');
                end
            end
            config = riesz_transform_object_perpare_filters(config);
        end
    end
end
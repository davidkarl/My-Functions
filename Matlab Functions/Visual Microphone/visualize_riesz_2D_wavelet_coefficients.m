function visualize_riesz_2D_wavelet_coefficients(riesz_transform_object, riesz_wavelet_coefficients, flag_grid)
%RIESZVISUALIZE2D display riesz-wavelet coefficients
%
% Display the Riesz-wavelet coefficients Q at each scale
%
% --------------------------------------------------------------------------
% Input arguments:
%
%
% CONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform.
%
% Q Riesz-wavelet coefficients to display 
%
% GRID show the coefficients from the same scale in the same figure
% with a grid layout if 1. Display coefficients in separate images if 0.
% Optional. Default is 1.
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if(nargin < 3)
    flag_grid = 1;
end

if flag_grid,
    number_of_columns = ceil(sqrt(riesz_transform_object.number_of_riesz_channels));
    number_of_rows = ceil(riesz_transform_object.number_of_riesz_channels/number_of_columns);
    for scale = 1:riesz_transform_object.number_of_scales,
        figure;
        for band = 1:riesz_transform_object.number_of_riesz_channels,
            subplot(number_of_rows, number_of_columns, band),
            imagesc(riesz_wavelet_coefficients{scale}(:,:,band));
            axis image; 
            axis off;
            title(sprintf('Scale %d -- Channel %d', scale, band));
        end
    end
    if (length(riesz_wavelet_coefficients) > riesz_transform_object.number_of_scales)
        figure;
        for band = 1:riesz_transform_object.number_of_riesz_channels,
            subplot(number_of_rows, number_of_columns, band),
            imagesc(riesz_wavelet_coefficients{riesz_transform_object.number_of_scales+1}(:,:,band)); 
            axis image; 
            axis off;
            title(sprintf('Coarse scale -- Channel %d',  band));
        end 
    end
else
    for scale = 1:riesz_transform_object.number_of_scales,
        for band = 1:riesz_transform_object.number_of_riesz_channels,
            figure; 
            imagesc(riesz_wavelet_coefficients{scale}(:,:,band));
            axis image; 
            axis off;
            title(sprintf('Scale %d -- Channel %d', scale, band));
        end
    end
    if (length(riesz_wavelet_coefficients) > riesz_transform_object.number_of_scales)
        for band = 1:riesz_transform_object.number_of_riesz_channels,
            figure; 
            imagesc(riesz_wavelet_coefficients{riesz_transform_object.number_of_scales+1}(:,:,band)); 
            axis image;
            axis off
            title(sprintf('Coarse scale -- Channel %d',  band));
        end
    end
    
end


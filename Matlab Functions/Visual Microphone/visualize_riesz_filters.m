function filters = visualize_riesz_filters(riesz_transform_object, generalization_matrix, ...
                                           visualization_zoom_factor, flag_same_figure, flag_save_tiff, flag_save_direcory)
%VISUALIZERIESZFILTERS View the filters for the specified Riesz transform.
%
% FILTERS = VISUALIZERIESZFILTERS(RIESZCONFIG2D, P, ZOOMINGFACTOR, SAVETIFF,SAVEDIR)
% display the filters for the Riesz transform specified by
% rieszConfig2D with generalization matrix P, zoomed with factor P. Images
% of filters are saved to the saveDir directory
%
% --------------------------------------------------------------------------
% Input arguments:
% 
% RIESZCONFIG2D configuration object of type RieszConfig2D for the Riesz
% transform
%
% P generalization matrix. Optional. Default is eye(rieszConfig2D.RieszChannels))
%
% ZOOMINGFACTOR: integer > 1 specifying the level interpolating zoom for
% the filter. Padding in Fourier domain is used for the interpolation.
% Optional. Default is 3.
%
% SAVETIFF: 1 if images of filters need to be save as .tiff files.
% Optional. Default is 0.
%
% SAVEDIR: directory where to save images of filters if SAVETIFF == 1.
% Optional. Default is 'rieszFilters'.
%


if (nargin < 2)
    generalization_matrix = eye(riesz_transform_object.number_of_riesz_channels);
end
if (nargin<3)
    visualization_zoom_factor = 3;
else
    visualization_zoom_factor = max(1, round(visualization_zoom_factor));
end
if (nargin<4)
    flag_same_figure = 1;
end
if (nargin< 5)
    flag_save_tiff = 0;
end
if (nargin < 6)
    flag_save_direcory = 'rieszFilters';
end

mask = riesz_transform_object.prefilter.filterLow;
filters = riesz_transform_object.riesz_transform_filters;
if (nargin>1)
    %use P to recombine filters
    filtersP = cell(1, length(filters));
    for i=1:riesz_transform_object.number_of_riesz_channels,
        f = zeros(riesz_transform_object.mat_size);
        for j=1:riesz_transform_object.number_of_riesz_channels,
            f = f + riesz_transform_object.riesz_transform_filters{j}*generalization_matrix(j,i);
        end
        filtersP{i}=f;
    end
    filters = filtersP;
    clear filtersP;
end

maxAbs = 0;
imList = cell(1, length(filters));
for i = 1:length(filters),
    f = filters{i};
    pad = visualization_zoom_factor*riesz_transform_object.mat_size; % use paddidng in the Fourier domain for zooming
    tmp = padarray(fftshift(f.*mask), pad);
    tmp = fftshift(tmp);
    tmp = real(fftshift(ifft2(tmp)));
    
    [h1 w1] = size(tmp);
    tmp = tmp((pad(1)+1):(h1-pad(1)), (pad(2)+1):(w1-pad(2)));
    maxAbs = max(maxAbs, max(abs(tmp(:))));
    imList{i} = tmp;
end

if nargout == 0,
    clear filters;
end

if flag_same_figure,
    epsilon = 10e-7;
    number_of_columns = ceil(sqrt(riesz_transform_object.number_of_riesz_channels));
    number_of_rows = ceil(riesz_transform_object.number_of_riesz_channels/number_of_columns);
    figure
    for k = 1:riesz_transform_object.number_of_riesz_channels,
        subplot(number_of_rows, number_of_columns, k);
        imagesc(imList{k}, [-maxAbs+epsilon maxAbs+epsilon]); 
        colormap(gray); 
        axis image; 
        axis off 
        title(sprintf('channel %d', k));
    end
else
    epsilon = 10e-7;
    for i=1:riesz_transform_object.number_of_riesz_channels,
        figure; 
        imagesc(imList{i}, [-maxAbs+epsilon maxAbs+epsilon]); 
        colormap(gray); 
        axis image; 
        axis off
        title(sprintf('channel %d', i));
    end
end

if flag_save_tiff,
    mkdir(flag_save_direcory);
    for i=1:riesz_transform_object.number_of_riesz_channels,
        filename = strcat('filter',num2str(i),'.tif');
        tmp = (imList{i}+maxAbs+epsilon)/(2*maxAbs);
        imwrite(uint8(round(255*tmp)), strcat(flag_save_direcory, '/',filename), 'tif');
    end
end
end
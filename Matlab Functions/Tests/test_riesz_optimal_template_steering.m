function [rotation_angles_cell_array, steered_riesz_wavelet_cell_array] = test_riesz_optimal_template_steering(mat_in, ...
                                                            number_of_scales, ...
                                                            riesz_transform_order, ...
                                                            template)
%DEMO_OPTIMALTEMPLATESTEERING steer coefficients to maximize the response of a template
%
%  [ANG, QMAX] = DEMO_OPTIMALTEMPLATESTEERING(A, NUMSCALES, ORDER, TEMPLATE) steer the
% Riesz-wavelet coefficients of order ORDER for the image A in order to maximize the
% response of the template TEMPLATE that is defines a linear combination
% of Riesz channels.
%
% --------------------------------------------------------------------------
% Input arguments:
%
% They are all optional.
% 
% A: input image. If none is provided the method buildDefaultImage is
% called to build one.
%
% NUMSCALES: number of wavelet scales for the decomposition. Default is
% 3.
%
% ORDER: order of the Riesz transform. Default is 2.
%
% TEMPLATE: array specifying a linear combination of Riesz channels. The
% method finds at each point the angle that maximizes this linear
% combination. Default is 1 for the first Riesz channels and 0 for the
% other ones.
%
% --------------------------------------------------------------------------
%
% Output arguments:
%
% ANG: angles that yield the maximization of the template
% QMAX : Riesz-wavelet coefficients after optimal steering
%
% --------------------------------------------------------------------------


%create a default image:
mat_in = double(imread('barbara.tif'));
mat_in = mat_in(1:256,1:256);
figure;
imagesc(mat_in);
axis image;
axis off;
colormap gray;
title('original image')

%setup the 2D Riesz transform:
number_of_scales = 3;
riesz_transform_order = 2;
smoothing_filter_sigma = 1.5;
flag_restrict_angle_values = 0;
template = zeros(1, riesz_transform_order +1); %maximize response of the first channel
template(1) = 1;


%perform a Riesz transform:
% prepare the transform
riesz_transform_object1 = riesz_transform_object(size(mat_in), riesz_transform_order, number_of_scales, 1);
% compute the Riesz-wavelet coefficients:
riesz_wavelet_cell_array = multiscale_riesz_analysis(mat_in, riesz_transform_object1);

%%  optimal template steering
% check if template corresponds to a single channel
if (length(find(abs(template))) == 1)
    channel = find(abs(template)); 
    [rotation_angles_cell_array, steered_riesz_wavelet_cell_array] = ...
                riesz_steer_channel_to_maximum_response(riesz_wavelet_cell_array, riesz_transform_object1, channel);
else
    [rotation_angles_cell_array, steered_riesz_wavelet_cell_array] = ...
                riesz_steer_template_to_maximum_response(riesz_wavelet_cell_array, riesz_transform_object1, template);
end

%% display angles at each scale
for scale_counter = 1:number_of_scales,
    %display angles
    figure;
    imagesc(rotation_angles_cell_array{scale_counter});
    colormap('hsv');
    axis off;
    axis image;
    title(sprintf('Template Steering - Angle at scale %d', scale_counter));
end

%% display template value after and before steering
for scale_counter = 1:number_of_scales,
    %display angles
    figure;
    % compute the original template image
    tmp = zeros(size(riesz_wavelet_cell_array{scale_counter}, 1), size(riesz_wavelet_cell_array{scale_counter}, 2));
    for k = 1:riesz_transform_object1.number_of_riesz_channels,
        tmp = tmp + template(k)*riesz_wavelet_cell_array{scale_counter}(:,:,k);
    end
    subplot(1,2,1); imagesc(tmp); axis off; axis image;
    title(sprintf('Original template coefficients at scale %d', scale_counter));
    subplot(1,2,2); imagesc(steered_riesz_wavelet_cell_array{scale_counter}); axis off; axis image;
    title(sprintf('Steered template coefficients at scale %d', scale_counter));
    clear tmp
end

end


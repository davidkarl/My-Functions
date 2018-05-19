function [rotation_angles_cell_array , coherency_cell_array] = ...
                             riesz_monogenic_analysis_of_riesz_coefficients2(riesz_wavelet_matrices_cell_array, ...
                                                                            riesz_transform_object1, ...
                                                                            smoothing_filter_sigma, ...
                                                                            flag_restrict_angle_value, ...
                                                                            mat_in_wavelet_cell_array)
%MONOGENICANALYSISOFRIESZCOEFFS perform the multiscale monogenic analysis of given sets
% of Riesz-wavelet coefficients
%
% --------------------------------------------------------------------------
% Input arguments:
%
%
% QA Riesz-wavelet coefficients used for the monogenic analysis
% 
% RIESZCONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform.
%
% SIGMA regularization parameter. It is the standard deviation of the regularizing
% Gaussian kernel.
%
% FULL Specifies if angles should be restricted to values in [-pi/2,pi/2].
% Optional, default is 0.
%
% A an image to analyze. It is required only if full == 1.
%
% --------------------------------------------------------------------------
% Output arguments:
%
% ANG angles estimated pointwise in the wavelet bands. It consists in a cell
% of matrices. Each element of the cell corresponds to the matrix of angles
% for a wavelet band.
%
% COHERENCY coherency values estimated pointwise in the wavelet bands.
% It consists in a cell of matrices. Each element of the cell corresponds to
% the matrix of coherency values for a wavelet band.
%


%riesz transform parameters:
riesz_transform_order = riesz_transform_object1.riesz_transform_order;
number_of_scales = riesz_transform_object1.number_of_scales;


if riesz_transform_order ~= 1,
    error('The monogenic analysis requires Riesz coefficients of order 1');
end

if nargin>3,
    if (flag_restrict_angle_value && nargin==4)
        error('Original image required to compute angles within an full range');
    end
else
    flag_restrict_angle_value = 0;
end

%Set number of sigmas before cutoff in filter:
smoothing_filter_number_of_sigmas_to_cutoff = 4; % Gaussian smoothing kernel cut = K*sigma

%Initialize angles and coherency cell arrays:
rotation_angles_cell_array = cell(1, number_of_scales);
coherency_cell_array = cell(1, number_of_scales);

%compute the regularization kernel:
smoothing_filter_axis_center = ceil(smoothing_filter_number_of_sigmas_to_cutoff*smoothing_filter_sigma) + 1;
smoothing_filter_width = 2*ceil(smoothing_filter_number_of_sigmas_to_cutoff*smoothing_filter_sigma) + 1;
smoothing_filter = zeros(smoothing_filter_width,smoothing_filter_width);
for x1 = 1:smoothing_filter_width;
    for x2 = 1:smoothing_filter_width;
        smoothing_filter(x1, x2) = exp(-((x1-smoothing_filter_axis_center)^2 + (x2-smoothing_filter_axis_center)^2)...
                                                        / (2*smoothing_filter_sigma^2));
    end
end
%normalize:
smoothing_filter = smoothing_filter/sum(smoothing_filter(:));

%% full range angle computation
mat_in_wavelet_gradient_angle = mat_in_wavelet_cell_array;
if flag_restrict_angle_value == 1, %compute sign of the direction thanks to the gradient of the wavelet coefficients
    
    %Compute gradient of wavelet coefficients for different scales:
    for scale_counter = 1:number_of_scales,
        %smooth current wavelet coefficients:
        %MAYBE ADD THAT INSTEAD OF IMFILTER USE A NEW FUNCTION WHICH DOES
        %convolve_without_end_effects BUT FOR 2D?!!?
        mat_in_wavelet_gradient_angle{scale_counter} = imfilter(mat_in_wavelet_gradient_angle{scale_counter}, ...
                                                                       smoothing_filter, 'symmetric');
        %Compute gradient for current scale wavelet:
        [FX,FY] = gradient(mat_in_wavelet_gradient_angle{scale_counter});
        
        %determine sign of the angle from the gradient:
        %(KIND OF WEIRD... WHY NOT ATAN2 AND THEN LOOK AT GRADIENT):
        mat_in_wavelet_gradient_angle{scale_counter} = atan2(FY, FX);
    end
end


%loop over the scales:
for scale_counter = 1:number_of_scales,
    %compute the 4 Jmn maps:
    if (size(riesz_wavelet_matrices_cell_array{scale_counter},3)==1) %ordinary wavelet transform (riesz order = 0)
        J11 = real(riesz_wavelet_matrices_cell_array{scale_counter}).^2;
        J12 = real(riesz_wavelet_matrices_cell_array{scale_counter}).*imag(riesz_wavelet_matrices_cell_array{scale_counter});
        J22 = imag(riesz_wavelet_matrices_cell_array{scale_counter}).^2;
    else
        J11 = riesz_wavelet_matrices_cell_array{scale_counter}(:,:,1).^2;
        J12 = riesz_wavelet_matrices_cell_array{scale_counter}(:,:,1).*riesz_wavelet_matrices_cell_array{scale_counter}(:,:,2);
        J22 = riesz_wavelet_matrices_cell_array{scale_counter}(:,:,2).^2;
    end
    
    %convolve the maps with the regularization kernel:
    J11 = imfilter(J11, smoothing_filter, 'symmetric');
    J12 = imfilter(J12, smoothing_filter, 'symmetric');
    J22 = imfilter(J22, smoothing_filter, 'symmetric');
    
    %compute coherency (UNDERSTAND WHY THIS IS COHERENCY!!!!):
    coherency_cell_array{scale_counter} = sqrt(((J22-J11).^2 + 4*J12.^2)) ./ (J22 + J11 + 2);
     
    %compute the first eigenvalue table (UNDERSTAND WHY THIS ENABLES PHASE CALCULATION!!!):
    lambda1 = ( J22 + J11 + sqrt((J11-J22).^2 + 4*J12.^2) ) / 2;
    
    if flag_restrict_angle_value, %use the gradient to discriminate angles shifted by pi:
        rotation_angles_cell_array{scale_counter} = atan((lambda1-J11)./J12) + pi*(mat_in_wavelet_gradient_angle{scale_counter}<0);
    else
        %compute the first eigen vector direction:
        rotation_angles_cell_array{scale_counter} = atan((lambda1-J11)./J12);
    end
end
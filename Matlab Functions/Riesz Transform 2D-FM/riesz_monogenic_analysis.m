function [rotation_angles_cell_array , coherency_cell_array] = riesz_monogenic_analysis(mat_in, ...
                                                                                        number_of_scales, ...
                                                                                        smoothing_filter_sigma, ...
                                                                                        flag_restrict_angle_value)
%MONOGENICANALYSIS perform the multiscale monogenic analysis of an image
%
% --------------------------------------------------------------------------
% Input arguments:
%
% A an image to analyze
% 
% J the number of scales for the primary wavelet transform
%
% SIGMA regularization parameter. It is the standard deviation of the regularizing
% Gaussian kernel. Optional. Default is 1.5;
%
% FULL Specifies if angles should be restricted to values in [-pi/2,pi/2].
% Optional. Default is 0.
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



if nargin < 3
    smoothing_filter_sigma = 1.5;
end
if nargin < 4,
    flag_restrict_angle_value = 0;
end
 
%configure the Riesz transform of order 1:
riesz_transform_order = 1;
riesz_transform_object1 = riesz_transform_object(size(mat_in), riesz_transform_order, number_of_scales, 1);

% compute the Riesz-wavelet coefficients:
riesz_wavelet_cell_array = multiscale_riesz_analysis(mat_in, riesz_transform_object1);

%% monogenic analysis
[rotation_angles_cell_array , coherency_cell_array] = ...
                    riesz_monogenic_analysis_of_riesz_coefficients(riesz_wavelet_cell_array, ...
                                                                   riesz_transform_object1, ...
                                                                   smoothing_filter_sigma, ...
                                                                   flag_restrict_angle_value, ...
                                                                   mat_in);
                                                            
                                                            
                                                            
                                                            
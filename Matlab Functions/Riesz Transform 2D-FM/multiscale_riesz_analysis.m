function [riesz_wavelet_matrices_cell_array , mat_in_highpassed] = ...
                                                        multiscale_riesz_analysis(mat_in, riesz_transform_object)
%MULTISCALERIESZANALYSIS performs the Riesz-wavelet decomposition of
% high order for a 2D image
%
% --------------------------------------------------------------------------
% Input arguments:
%
% A image to analyze
%
% CONFIG RieszConfig2D object that specifies the primary Riesz-wavelet
% transform.
% 
% --------------------------------------------------------------------------
% Output arguments:
%
% Q structure containing the Riesz-wavelet coefficients. It consists in a
% cell of matrices. Each element of the cell corresponds to a wavelet
% scale. Each matrix is a 3D matrix whose 3rd dimension corresponds to
% Riesz channels.
%
% RESIDUAL high-pass residual band for the primary wavelet pyramid
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
% 
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

%% prefilter images
[mat_in_lowpassed , mat_in_highpassed] = riesz_prefilter(mat_in,riesz_transform_object);
%% Apply the multiscale Riesz transform
riesz_coefficients_matrix_of_lowpassed_image_3D = riesz_transform_analysis(mat_in_lowpassed,riesz_transform_object);
%% apply wavelet decomposition
riesz_wavelet_matrices_cell_array = wavelet_analysis_riesz_matrix(riesz_coefficients_matrix_of_lowpassed_image_3D, ...
                                                                  riesz_transform_object);
end
function Q1 = riesz_rotate_2D_riesz_coefficients(riesz_wavelet_matrices_cell_array, ...
                                                 riesz_transform_order, ...
                                                 rotation_angles_cell_array)
%ROTATE2DRIESZCOEFFS steer Riesz-wavelet coefficients
%
% --------------------------------------------------------------------------
% Input arguments:
%
% Q Riesz-wavelet coefficients
%
% ORDER the order of the Riesz transform
% 
% ACI: angles of rotation for the different scales. It consists in a structure
% of 2D matrices. Each matrix corresponds to the pointwise rotation angles for
% a given wavelet scale.
%
% --------------------------------------------------------------------------
% Output arguments:
%
% Q1 Riesz-wavelet coefficients steered according to the angles ACI
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

Q1 = riesz_wavelet_matrices_cell_array;
%compute each rotation matrix without polynomial coefficients pre-computation
for i = 1:length(rotation_angles_cell_array)
    sub = reshape(riesz_wavelet_matrices_cell_array{i}, ...
                        size(riesz_wavelet_matrices_cell_array{i},1) * size(riesz_wavelet_matrices_cell_array{i},2), ...
                        size(riesz_wavelet_matrices_cell_array{i}, 3));
    ang = rotation_angles_cell_array{i}(:);
    S = riesz_compute_multiple_rotation_matrices_for_2D_riesz_transform(ang, riesz_transform_order);
    for sample_counter = 1:size(ang, 1)
        sub(sample_counter, :) = (S(:,:, sample_counter)*sub(sample_counter, :)')';
    end
    Q1{i} = reshape(sub, size(riesz_wavelet_matrices_cell_array{i}, 1), ...
                         size(riesz_wavelet_matrices_cell_array{i}, 2), ...
                         size(riesz_wavelet_matrices_cell_array{i}, 3));
end



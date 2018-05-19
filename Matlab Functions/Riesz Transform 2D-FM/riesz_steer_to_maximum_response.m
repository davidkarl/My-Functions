function [rotation_angles_cell_array, steered_riesz_wavelet_cell_array] = ...
                       riesz_steer_to_maximum_response(riesz_wavelet_cell_array, ...
                                                       riesz_transform_object1, ...
                                                       channel_to_maximize)
% MAXSTEER steer Riesz coefficients to maximize the response in a given channel
%
% --------------------------------------------------------------------------
% Input arguments:
%
% Q structure of Riesz-wavelet coefficients
%
% CONFIG RieszConfig2D object that specifies the primary Riesz-wavelet
% transform.
% 
% CHANNEL Riesz channel for which the response is to be maximized
%
% --------------------------------------------------------------------------
% Output arguments:
%
% ANG angles estimated pointwise in the wavelet bands. It consists in a cell
% of matrices. Each element of the cell corresponds to the matrix of angles
% for a wavelet band.
%
% QMAX Riesz-wavelet coefficient steered with respect to the angles in ANG
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

number_of_scales = riesz_transform_object1.number_of_scales;
riesz_transform_order = riesz_transform_object1.riesz_transform_order;

rotation_angles_cell_array = cell(1,number_of_scales);
if nargout >1,
    steered_riesz_wavelet_cell_array = cell(1, number_of_scales+1);
    steered_riesz_wavelet_cell_array{number_of_scales+1} = riesz_wavelet_cell_array{number_of_scales+1};
end

for scale_counter = 1:number_of_scales,
    if nargout >1,
        [rotation_angles_cell_array{scale_counter}, steered_riesz_wavelet_cell_array{scale_counter}] = ...
                            riesz_get_angles_that_give_maximum_response(riesz_wavelet_cell_array{scale_counter}, ...
                                                                        riesz_transform_order, ...
                                                                        channel_to_maximize);
    else
        rotation_angles_cell_array{scale_counter} = ...
                            riesz_get_angles_that_give_maximum_response(riesz_wavelet_cell_array{scale_counter}, ...
                                                                        riesz_transform_order, ...
                                                                        channel_to_maximize);
    end
end

end

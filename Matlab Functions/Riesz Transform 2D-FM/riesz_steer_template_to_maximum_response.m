function [ang, Qmax] = riesz_steer_template_to_maximum_response(riesz_wavelet_cell_array, ...
                                                                riesz_transform_object1, ...
                                                                template)
%MAXSTEERTEMPLATE steer Riesz coefficients to maximize the response of a template
%
% --------------------------------------------------------------------------
% Input arguments:
%
% Q structure of Riesz-wavelet coefficients
%
% RIESZCONFIG RieszConfig2D object that specifies the primary Riesz-wavelet
% transform.
% 
% TEMPLATE coefficients for the linear combination of Riesz channels for which
% the response is to be maximized
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

ang = cell(1,number_of_scales);
if nargout >1,
    Qmax = cell(1, number_of_scales+1);
    Qmax{number_of_scales+1} = riesz_wavelet_cell_array{number_of_scales+1};
end
for i=1:number_of_scales,
    if nargout >1,
        [ang{i}, Qmax{i}] = riesz_angle_template(riesz_wavelet_cell_array{i}, template, riesz_transform_order);
    else
        ang{i} = riesz_angle_template(riesz_wavelet_cell_array{i}, template, riesz_transform_order);
    end
end
function y = wavelet_interface_wavelab_ti(x, Jmin, qmf)
% wavelab TI interface

if iscell(x)
    dir = -1; ndim = get_actual_number_of_dimensions(x{1});
else
    dir = 1; ndim = get_actual_number_of_dimensions(x);
end

if ndim==1
    if dir==1
        y = wavelet_convert_translation_invariant_to_stationary_transform( wavelet_translation_invariant_forward_transform(x,Jmin,qmf) );
        %transform into cell array:
        y = transform_matrix_into_cell_array(y);
    else
        x = transform_matrix_into_cell_array(x);
        y = wavelet_invert_translation_invariant_wavelet_transform( ...
                                    wavelet_convert_stationary_to_translation_invariant_transform(x),qmf);
        y = y(:);
    end
elseif ndim==2
    if dir==1
        y = wavelet_2D_translation_invariant_forward_transform(x,Jmin,qmf);
        n = size(x,1);
        y = reshape( y, [n size(y,1)/n n] );
        y = permute(y, [1 3 2]);
        % transform into cell array
        y = transform_matrix_into_cell_array(y);
    else
        x = transform_matrix_into_cell_array(x);
        x = permute(x, [1 3 2]);
        x = reshape( x, [size(x,1)*size(x,2) size(x,3)] );
        y = wavelet_2D_invert_translation_invariant_wavelet_transform(x,Jmin,qmf);
    end
end
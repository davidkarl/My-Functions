function y = wavelet_interface_lifting(vec_in, Jmin, number_of_vanishing_moments)
% lifting interface

if number_of_vanishing_moments==2
    options.filter = 'linear';
else
    options.filter = '9-7';
end
options.ti = 1;

direction = 1;
if iscell(vec_in)
    direction = -1;
end

if direction<0
    vec_in = { vec_in{end} vec_in{end-1:-1:1} };
    vec_in = transform_matrix_into_cell_array(vec_in);
    if size(vec_in,3)==1
        % 1D
        vec_in = reshape(vec_in, [size(vec_in,1) 1 size(vec_in,2)]);
    end
end
y = perform_lifting_transform(vec_in, Jmin, direction, options);
if direction>0
    y = transform_matrix_into_cell_array(squeeze(y));
    % reverse frequencies
    y = { y{end:-1:2} y{1} };
end
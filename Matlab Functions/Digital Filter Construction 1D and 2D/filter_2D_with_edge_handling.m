function y = filter_2D_with_edge_handling(mat_in, filter_in, extension_mode_string, shift_vec)
% EFILTER2   2D Filtering with edge handling (via extension)
%
%	y = efilter2(x, f, [extmod], [shift])
%
% Input:
%	x:	input image
%	f:	2D filter
%	extmod:	[optional] extension mode (default is 'per')
%	shift:	[optional] specify the window over which the 
%		convolution occurs. By default shift = [0; 0].
%
% Output:
%	y:	filtered image that has:
%		Y(z1,z2) = X(z1,z2)*F(z1,z2)*z1^shift(1)*z2^shift(2)
%
% Note:
%	The origin of filter f is assumed to be floor(size(f)/2) + 1.
%	Amount of shift should be no more than floor((size(f)-1)/2).
%	The output image has the same size with the input image.
%
% See also:	EXTEND2, SEFILTER2

if ~exist('extension_mode_string', 'var')
    extension_mode_string = 'per';
end

if ~exist('shift_vec', 'var')
    shift_vec = [0; 0];
end

% Periodized extension:
filter_in_center = (size(filter_in) - 1) / 2;
max_filter_size = max(size(filter_in));
max_mat_in_size = max(size(mat_in));
%if filter is to big (bigger then image) then cut it down to size:
if max_filter_size > max_mat_in_size
    filter_in = fit_matrix_dimensions_to_certain_size(filter_in, max(size(mat_in)) - 1);
    filter_in_center = (size(filter_in) - 1) / 2;
end


%Pad mat_in to account for filter as well:
mat_in_extended = extend_2D(mat_in, ...
                            floor(filter_in_center(1)) + shift_vec(1), ...
                            ceil(filter_in_center(1)) - shift_vec(1), ...
                            floor(filter_in_center(2)) + shift_vec(2), ...
                            ceil(filter_in_center(2)) - shift_vec(2), ...
                            extension_mode_string);

% y = fconv2(xext, f, 'valid');

if max_filter_size < 50
    % Convolution and keep the central part that has the size as the input
    y = conv2(mat_in_extended, filter_in, 'valid');
else
    y = conv_fft_2D(mat_in_extended, filter_in, 'valid');
end


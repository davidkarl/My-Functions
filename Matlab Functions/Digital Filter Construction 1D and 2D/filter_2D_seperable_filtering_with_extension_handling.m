function y = filter_2D_seperable_filtering_with_extension_handling(mat_in, filter1, filter2, extension_mode_string, shift_vec)
% SEFILTER2   2D seperable filtering with extension handling
%
%       y = sefilter2(x, f1, f2, [extmod], [shift])
%
% Input:
%   x:      input image
%   f1, f2: 1-D filters in each dimension that make up a 2D seperable filter
%   extmod: [optional] extension mode (default is 'per')
%   shift:  [optional] specify the window over which the 
%       	convolution occurs. By default shift = [0; 0].
%
% Output:
%   y:      filtered image of the same size as the input image:
%           Y(z1,z2) = X(z1,z2)*F1(z1)*F2(z2)*z1^shift(1)*z2^shift(2)
%
% Note:
%   The origin of the filter f is assumed to be floor(size(f)/2) + 1.
%   Amount of shift should be no more than floor((size(f)-1)/2).
%   The output image has the same size with the input image.
%
% See also: EXTEND2, EFILTER2

if ~exist('extension_mode_string', 'var')
    extension_mode_string = 'per';
end

if ~exist('shift_vec', 'var')
    shift_vec = [0; 0];
end

% Make sure filter in a row vector
filter1 = filter1(:)';
filter2 = filter2(:)';

% Periodized extension
lf1 = (length(filter1) - 1) / 2;
lf2 = (length(filter2) - 1) / 2;

y = extend_2D(mat_in, floor(lf1) + shift_vec(1), ceil(lf1) - shift_vec(1), ...
    floor(lf2) + shift_vec(2), ceil(lf2) - shift_vec(2), extension_mode_string);

% Seperable filter
y = conv2(filter1, filter2, y, 'valid');
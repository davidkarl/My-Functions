mat_in = create_speckles_of_certain_size_in_pixels(50,512,1,1);
mat_in = abs(mat_in).^2;
mat_in_temp = mat_in;
mat_in = mat_in(1:400,1:400);
mat_in_temp = mat_in_temp(1:400,1:400);
figure(1);
imagesc(mat_in_temp);

filter_in = fspecial('gaussian',10);

%(1). convolution 2D ft:
%assuming both matrices are rectangular AND with even number of parts,
%i only return a mat of size equal to mat1,
flag_use_padding = 1;
if flag_use_padding==1 
    [mat_in,filter_in,indices] = pad_arrays_for_convolution(mat_in,filter_in,1);
end
spacing = 1;
U1 = ft2(mat_in, spacing); % DFTs of signals: fftshift(fft2(fftshift(mat_in)))*spacing^2
U2 = ft2(filter_in, spacing);
delta_f = 1/(size(mat_in,1)*spacing); % frequency grid spacing [m]
convolved_mat = real(ift2(U1 .* U2, delta_f));
if flag_use_padding == 1
    % trim to correct output size
    convolved_mat = convolved_mat(indices(1):indices(2),indices(3):indices(4)); 
end
figure(2);
imagesc(convolved_mat);
title('ift2(ft2(mat1).*ft2(mat2), convolution 2D ft');


%(2). conv fft 2D: 
mat_in = mat_in_temp;
filter_in = fspecial('gaussian',10);
shape = 'valid';
if ~exist('shape', 'var')
    shape = 'full';
end
%take size x f:
[mat_in_rows,mat_in_columns] = size(mat_in);
[filter_in_rows,filter_in_columns] = size(filter_in);
%number of rows and columns to use in fft (power of 2):
M = 2^nextpow2(mat_in_rows+filter_in_rows-1); 
N = 2^nextpow2(mat_in_columns+filter_in_columns-1);
%Perform covolution in fft domain:
y = ifft2(fft2(mat_in,M,N).*fft2(filter_in,M,N));
switch shape
    case {'full'}
        y = y(1:mat_in_rows+filter_in_rows-1,1:mat_in_columns+filter_in_columns-1);
    case {'same'}
        mbh = ceil((filter_in_rows+1)/2); 
        nbh = ceil((filter_in_columns+1)/2);
        y = y(mbh:mbh+mat_in_rows-1,nbh:nbh+mat_in_columns-1);
    case {'valid'}
        y = y(filter_in_rows:mat_in_rows,filter_in_columns:mat_in_columns);
    otherwise
        disp('unvalid shape');
end
figure(3);
imagesc(y);
title('conv fft 2D, shape = valid');

%(3). conv2 zeros boundary conditions:
if ~exist('flag_center_origin','var')
  flag_center_origin = 0;
end
%Check that one mat is absolutely larger than the other:
if ( size(mat_in,1) >= size(filter_in,1) && size(mat_in,2) >= size(filter_in,2) )
    large_mat = mat_in; 
    small_mat = filter_in;
elseif  (( size(mat_in,1) <= size(filter_in,1) ) && ( size(mat_in,2) <= size(filter_in,2) ))
    large_mat = filter_in; 
    small_mat = mat_in;
else
  error('one arg must be larger than the other in both dimensions!');
end
%Get mat sizes:
large_mat_rows = size(large_mat,1);
large_mat_columns = size(large_mat,2);
small_mat_rows = size(small_mat,1);
small_mat_columns = size(small_mat,2);
%These values are the index of the small mtx that falls on the
%border pixel of the large matrix when computing the first convolution response sample:
small_mat_center_rows = floor((small_mat_rows+flag_center_origin+1)/2);
small_mat_center_columns = floor((small_mat_columns+flag_center_origin+1)/2);
%Filter:
clarge = conv2(large_mat,small_mat);
filtered_mat = clarge(small_mat_center_rows:large_mat_rows+small_mat_center_rows-1, ...
                      small_mat_center_columns:large_mat_columns+small_mat_center_columns-1);


%(4). conv2 reflective boundary conditions:
if ~exist('flag_filter_origin_location','var')
  flag_filter_origin_location = 0;
end
if (size(mat_in,1) >= size(filter_in,1)) && (size(mat_in,2) >= size(filter_in,2))
    large_mat = mat_in; 
    small_mat = filter_in;
elseif  (size(mat_in,1) <= size(filter_in,1)) && (size(mat_in,2) <= size(filter_in,2))
    large_mat = filter_in; 
    small_mat = mat_in;
else
    %WHAT?! HARTA!
    error('one arg must be larger than the other in both dimensions!');
end
%Get mat sizes:
large_mat_rows = size(large_mat,1);
large_mat_columns = size(large_mat,2);
small_mat_rows = size(small_mat,1);
small_mat_columns = size(small_mat,2);
%These values are one less than the index of the small mtx that falls on 
%the border pixel of the large matrix when computing the first convolution response sample:
small_mat_row_center = floor( (small_mat_rows+flag_filter_origin_location-1)/2 );
small_mat_column_center = floor( (small_mat_columns+flag_filter_origin_location-1)/2 );
small_mat_shifted_row_center = small_mat_rows - small_mat_row_center;
small_mat_shifted_column_center = small_mat_columns - small_mat_column_center;
large_mat_valid_rows = large_mat_rows - small_mat_row_center;
large_mat_valid_columns = large_mat_columns - small_mat_column_center;
%Pad with reflected copies:
large_mat_padded = [ 
    large_mat(small_mat_shifted_row_center:-1:2, small_mat_shifted_column_center:-1:2), ...
    large_mat(small_mat_shifted_row_center:-1:2, :), ...
	large_mat(small_mat_shifted_row_center:-1:2, large_mat_columns-1:-1:large_mat_valid_columns)...
    ; ...
    large_mat(:, small_mat_shifted_column_center:-1:2),    ...
    large_mat,   ...
    large_mat(:, large_mat_columns-1:-1:large_mat_columns-small_mat_column_center)...
    ; ...
    large_mat(large_mat_rows-1:-1:large_mat_valid_rows, small_mat_shifted_column_center:-1:2), ...
    large_mat(large_mat_rows-1:-1:large_mat_valid_rows, :), ...
    large_mat(large_mat_rows-1:-1:large_mat_valid_rows, large_mat_columns-1:-1:large_mat_valid_columns) ...
    ];
%Actually convolve mats:
filtered_mat = conv2(large_mat_padded,small_mat,'valid');
figure(4);
imagesc(large_mat_padded);
title('reflective boundary conditions padded mat');
figure(5);
imagesc(filtered_mat);
title('reflective boundary conditions, shape=valid');


%(5). conv2 circular convolution:
if ~exist('flag_center_origin','var')
    flag_center_origin = 0;
end
if ( size(mat_in,1) >= size(filter_in,1)  && size(mat_in,2) >= size(filter_in,2) )
    large_mat = mat_in; 
    small_mat = filter_in;
elseif  (( size(mat_in,1) <= size(filter_in,1) ) && ( size(mat_in,2) <= size(filter_in,2) ))
    large_mat = filter_in; 
    small_mat = mat_in;
else
    error('one arg must be larger than the other in both dimensions!');
end
large_mat_rows = size(large_mat,1);
large_mat_columns = size(large_mat,2);
small_mat_rows = size(small_mat,1);
small_mat_columns = size(small_mat,2);
% These values are the index of the small mtx that falls on the
% border pixel of the large matrix when computing the first
% convolution response sample:
small_mat_row_center = floor((small_mat_rows+flag_center_origin+1)/2);
small_mat_column_center = floor((small_mat_columns+flag_center_origin+1)/2);
%pad:
clarge = [ ...
    large_mat(large_mat_rows-small_mat_rows+small_mat_row_center+1:large_mat_rows,large_mat_columns-small_mat_columns+small_mat_column_center+1:large_mat_columns), large_mat(large_mat_rows-small_mat_rows+small_mat_row_center+1:large_mat_rows,:), ...
    large_mat(large_mat_rows-small_mat_rows+small_mat_row_center+1:large_mat_rows,1:small_mat_column_center-1); ...
    large_mat(:,large_mat_columns-small_mat_columns+small_mat_column_center+1:large_mat_columns), large_mat, large_mat(:,1:small_mat_column_center-1); ...
    large_mat(1:small_mat_row_center-1,large_mat_columns-small_mat_columns+small_mat_column_center+1:large_mat_columns), ...
    large_mat(1:small_mat_row_center-1,:), ...
    large_mat(1:small_mat_row_center-1,1:small_mat_column_center-1) ];
c = conv2(clarge,small_mat,'valid');
figure(6);
imagesc(clarge);
title('circular boundary conditions padded mat');
figure(10);
imagesc(c);
title('circular boundary conditions shape=valid');


% %(6). filter 2D seperable filtering with extention handling:
% % Input:
% %   x:      input image
% %   f1, f2: 1-D filters in each dimension that make up a 2D seperable filter
% %   extmod: [optional] extension mode (default is 'per')
% %   shift:  [optional] specify the window over which the 
% %       	convolution occurs. By default shift = [0; 0].
% %
% % Output:
% %   y:      filtered image of the same size as the input image:
% %           Y(z1,z2) = X(z1,z2)*F1(z1)*F2(z2)*z1^shift(1)*z2^shift(2)
% %
% % Note:
% %   The origin of the filter f is assumed to be floor(size(f)/2) + 1.
% %   Amount of shift should be no more than floor((size(f)-1)/2).
% %   The output image has the same size with the input image.
% if ~exist('extension_mode_string', 'var')
%     extension_mode_string = 'per';
% end
% if ~exist('shift', 'var')
%     shift_vec = [0; 0];
% end
% %Make sure filter in a row vector:
% filter1 = filter1(:)';
% filter2 = filter2(:)';
% %Periodized extension:
% lf1 = (length(filter1) - 1) / 2;
% lf2 = (length(filter2) - 1) / 2;
% y = extend_2D(mat_in, floor(lf1) + shift_vec(1), ceil(lf1) - shift_vec(1), ...
%     floor(lf2) + shift_vec(2), ceil(lf2) - shift_vec(2), extension_mode_string);
% % Seperable filter
% y = conv2(filter1, filter2, y, 'valid');



%(7). filter 2D with edge handling:
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
if ~exist('extension_mode_string', 'var')
    extension_mode_string = 'per';
end
if ~exist('shift_vec', 'var')
    shift_vec = [0; 0];
end
%Periodized extension:
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
if max_filter_size < 50
    % Convolution and keep the central part that has the size as the input
    y = conv2(mat_in_extended, filter_in, 'valid');
else
    y = conv_fft_2D(mat_in_extended, filter_in, 'valid');
end
figure(3);
subplot(2,1,1);
imagesc(mat_in);
title('original mat');
subplot(2,1,2);
imagesc(mat_in_extended);
title('mat in extended');




%%%%  EXTEND 2D %%%%%
% EXTEND2   2D extension
%
%	y = extend2(x, ru, rd, cl, cr, extmod)
%
% Input:
%	x:	input image
%	ru, rd:	amount of extension, up and down, for rows
%	cl, cr:	amount of extension, left and rigth, for column
%	extmod:	extension mode.  The valid modes are:
%       'sym':		symmetric extension (both direction)
%		'per':		periodized extension (both direction)
%		'qper_row':	quincunx periodized extension in row
%		'qper_col':	quincunx periodized extension in column
%   sympopt : optional argument for symmetric extension, affect the center 
%   of symmetry at the four borders of matrix 
%   0 : the border is repeated, 
%   1 : the border is not repeated, the second row (column) is repeated
%   default [1,1,1,1]
figure(1);
imagesc(mat_in);
title('mat in');
mat_in_extended1 = extend_2D(mat_in, ...
                            floor(filter_in_center(1)) + shift_vec(1), ...
                            ceil(filter_in_center(1)) - shift_vec(1), ...
                            floor(filter_in_center(2)) + shift_vec(2), ...
                            ceil(filter_in_center(2)) - shift_vec(2), ...
                            'sym');
figure(2);
imagesc(mat_in_extended1);
title('sym');
mat_in_extended2 = extend_2D(mat_in, ...
                            floor(filter_in_center(1)) + shift_vec(1), ...
                            ceil(filter_in_center(1)) - shift_vec(1), ...
                            floor(filter_in_center(2)) + shift_vec(2), ...
                            ceil(filter_in_center(2)) - shift_vec(2), ...
                            'per');
figure(3);
imagesc(mat_in_extended2);
title('per');
mat_in_extended3 = extend_2D(mat_in, ...
                            floor(filter_in_center(1)) + shift_vec(1), ...
                            ceil(filter_in_center(1)) - shift_vec(1), ...
                            floor(filter_in_center(2)) + shift_vec(2), ...
                            ceil(filter_in_center(2)) - shift_vec(2), ...
                            'qper_row');
figure(4);
imagesc(mat_in_extended3);
title('qper row');
mat_in_extended4 = extend_2D(mat_in, ...
                            floor(filter_in_center(1)) + shift_vec(1), ...
                            ceil(filter_in_center(1)) - shift_vec(1), ...
                            floor(filter_in_center(2)) + shift_vec(2), ...
                            ceil(filter_in_center(2)) - shift_vec(2), ...
                            'qper_col');
figure(5);
imagesc(mat_in_extended4);
title('qper col');






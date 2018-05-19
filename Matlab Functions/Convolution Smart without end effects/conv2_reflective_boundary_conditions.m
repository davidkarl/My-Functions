function [filtered_mat] = conv2_reflective_boundary_conditions(mat1,mat2,flag_filter_origin_location)

% RES = RCONV2(MTX1, MTX2, CTR)
%
% Convolution of two matrices, with boundaries handled via reflection
% about the edge pixels.  Result will be of size of LARGER matrix.
% 
% The origin of the smaller matrix is assumed to be its center.
% For even dimensions, the origin is determined by the CTR (optional) 
% argument:
%      CTR   origin
%       0     DIM/2      (default)
%       1     (DIM/2)+1  


if ~exist('flag_filter_origin_location','var')
  flag_filter_origin_location = 0;
end

if (size(mat1,1) >= size(mat2,1)) && (size(mat1,2) >= size(mat2,2))
    large_mat = mat1; 
    small_mat = mat2;
elseif  (size(mat1,1) <= size(mat2,1)) && (size(mat1,2) <= size(mat2,2))
    large_mat = mat2; 
    small_mat = mat1;
else
    %WHAT?! HARTA!
    error('one arg must be larger than the other in both dimensions!');
end

%Get mat sizes:
large_mat_rows = size(large_mat,1);
large_mat_columns = size(large_mat,2);
small_mat_rows = size(small_mat,1);
small_mat_columns = size(small_mat,2);

% These values are one less than the index of the small mtx that falls on 
% the border pixel of the large matrix when computing the first convolution response sample:
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


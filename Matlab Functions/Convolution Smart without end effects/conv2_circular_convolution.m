function c = conv2_circular_convolution(mat1,mat2,flag_center_origin)
% RES = CCONV2(MTX1, MTX2, CTR)
%
% Circular convolution of two matrices.  Result will be of size of
% LARGER vector.
%
% The origin of the smaller matrix is assumed to be its center.
% For even dimensions, the origin is determined by the CTR (optional)
% argument:
%      CTR   origin
%       0     DIM/2      (default)
%       1     (DIM/2)+1

if ~exist('flag_center_origin','var')
    flag_center_origin = 0;
end

if ( size(mat1,1) >= size(mat2,1)  && size(mat1,2) >= size(mat2,2) )
    large_mat = mat1; 
    small_mat = mat2;
elseif  (( size(mat1,1) <= size(mat2,1) ) && ( size(mat1,2) <= size(mat2,2) ))
    large_mat = mat2; 
    small_mat = mat1;
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


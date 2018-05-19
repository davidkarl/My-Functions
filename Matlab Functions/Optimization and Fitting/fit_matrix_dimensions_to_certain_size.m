function X = fit_matrix_dimensions_to_certain_size(mat_in,destination_matrix_size)
% FITMAT   fit a matrix to matrix of size N, truncate or expand if necessary
% Syntax 	
%   X = fitmat(Y,N)
% Input:
%   Y : the matrix to expand or cut
%   N : size of destination matrix , N*N if N is one value
%
% Output:
%   X:	Resulting matrix
%
% Note:
%   See formulat.tex in Tex folder
%      
% See also: 
% History 
% Apr,11,2004 : Creation ...
% Aug,15,2004 : Debug ...


if length(destination_matrix_size) == 1
    destination_matrix_size = [destination_matrix_size,destination_matrix_size];
end

test = mat_in - rot90(mat_in,2);
test = sum(test(:));

if (test > 10^(-5))
    disp('NOTE, Y not symetric');
%     return;
end

mat_in_size = size(mat_in);
new_mat_size = ...
        [ max(mat_in_size(1),destination_matrix_size(1)) , max(mat_in_size(2),destination_matrix_size(2)) ];
Xbig = zeros(new_mat_size);

diff = fix((new_mat_size-mat_in_size)/2);

Xbig(1+diff(1):diff(1)+mat_in_size(1) , 1+diff(2):diff(2)+mat_in_size(2)) = mat_in;

X = zeros(destination_matrix_size);

diff = fix((new_mat_size-destination_matrix_size)/2);

X = Xbig( 1+diff(1):diff(1)+destination_matrix_size(1) , 1+diff(2):diff(2)+destination_matrix_size(2) );



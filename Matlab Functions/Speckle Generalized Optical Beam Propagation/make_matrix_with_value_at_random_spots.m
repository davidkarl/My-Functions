function [matrix] = make_matrix_with_value_at_random_spots(N,M,number_of_cells,value,flag_use_background_matrix,background_matrix_value)
% N=10;
% M=10;
% number_of_cells=5;
% value=1;

flag=1;
total_size = N*M;
mat_cells = randperm(total_size);
mat_cells = mat_cells(1:number_of_cells);

matrix = zeros(N*M,1);
matrix = matrix(:);
matrix(mat_cells) = value;
matrix = reshape(matrix,[N,M]);

if flag_use_background_matrix==1
   matrix = matrix + ones(N,M).*background_matrix_value; 
end
% imagesc(matrix);
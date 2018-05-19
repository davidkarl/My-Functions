function [quantized_matrix] = quantize_matrix(matrix,number_of_bits,minimum_allowed_value,maximum_allowed_value,flag_precise_or_quick)
%number_of_quantization_levels = 2^(number_of_bits)-1

% matrix=randn(4,4);
% number_of_bits = 4;
% minimum_allowed_value=-1;
% maximum_allowed_value=1;


%get matrix size:
matrix_size = size(matrix);

if flag_precise_or_quick==1
    %set partition:
    partition = linspace(minimum_allowed_value,maximum_allowed_value,2^number_of_bits-1);

    %set codebook:
    codebook = linspace(minimum_allowed_value,maximum_allowed_value,2^number_of_bits);

    %quantize matrix:
    [index,quantized_matrix] = quantiz(matrix(:),partition,codebook);

else
    values_vec = linspace(minimum_allowed_value,maximum_allowed_value,2^number_of_bits);
    quantized_matrix = round2x(matrix(:),values_vec,'round');
end

quantized_matrix = reshape(quantized_matrix,matrix_size);



function wavelet_coefficients = wavelet_2D_transform_symmetric_extention_biorthogonal(mat_in,L_coarsest_scale,QMF,dual_QMF)
% FWT2_SBS -- 2-dimensional wavelet transform
%              (symmetric extension, bi-orthogonal)
%  Usage
%    wc = FWT2_SBS(x,L,qmf,dqmf)
%  Inputs
%    x     2-d image (n by n array, n arbitrary)
%    L     coarsest level
%    qmf   low-pass quadrature mirror filter
%    dqmf  high-pass dual quadrature mirror filter
%  Output
%    wc    2-d wavelet transform
%  Description
%    A two-dimensional Wavelet Transform is computed for the
%    matrix x. To reconstruct, use IWT2_SBS.
%  See Also
%    IWT2_SBS
%

[m_row_size,J_dyadic_length_rows] = dyadlength(mat_in(:,1));
[n_column_size,K_dyadic_length_columns] = dyadlength(mat_in(1,:));
wavelet_coefficients = mat_in;
row_size_current = m_row_size;
column_size_current = n_column_size;

J_dyadic_length_rows = min([J_dyadic_length_rows,K_dyadic_length_columns]);

for j_current_scale = J_dyadic_length_rows-1 : -1 : L_coarsest_scale

    %Get appropriate indices:
    if rem(row_size_current,2) == 0
        top_indices = (row_size_current/2+1):row_size_current;
        bottom_indices = 1:(row_size_current/2);
    else
        top_indices = ((row_size_current+1)/2+1):row_size_current;
        bottom_indices = 1:((row_size_current+1)/2);
    end
    if rem(column_size_current,2) == 0
        right_indices = (column_size_current/2+1):column_size_current;
        left_indices = 1:(column_size_current/2);
    else
        right_indices = ((column_size_current+1)/2+1):column_size_current;
        left_indices = 1:((column_size_current+1)/2);
    end

    %Get wavelet coefficients:
    for row_counter = 1:row_size_current
        row = wavelet_coefficients(row_counter,1:column_size_current);
        [beta,alpha] = wavelet_downsample_symmetric_dual_filter(row,QMF,dual_QMF);
        wavelet_coefficients(row_counter,left_indices) = beta;
        wavelet_coefficients(row_counter,right_indices) = alpha;
    end
    for column_counter = 1:column_size_current
        column = wavelet_coefficients(1:row_size_current,column_counter)';
        [beta,alpha] = wavelet_downsample_symmetric_dual_filter(column,QMF,dual_QMF);
        wavelet_coefficients(bottom_indices,column_counter) = beta';
        wavelet_coefficients(top_indices,column_counter) = alpha';
    end
    row_size_current = bottom_indices(length(bottom_indices));
    column_size_current = left_indices(length(left_indices));
end
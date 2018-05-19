function mat_in_transform = wavelet_2D_transform_orthogonal(mat_in,L_coarsest_level,QMF)
% FWT2_PO -- 2-d MRA wavelet transform (periodized, orthogonal)
%  Usage
%    wc = FWT2_PO(x,L,qmf)
%  Inputs
%    x     2-d image (n by n array, n dyadic)
%    L     coarse level
%    qmf   quadrature mirror filter
%  Outputs
%    wc    2-d wavelet transform
%
%  Description
%    A two-dimensional Wavelet Transform is computed for the
%    array x.  To reconstruct, use IWT2_PO.
%
%  See Also
%    IWT2_PO, MakeONFilter
%
[signal_length,J_dyadic_length] = quadlength(mat_in);
mat_in_transform = mat_in;
current_signal_length = signal_length;
for j_current_scale = J_dyadic_length-1 : -1 : L_coarsest_level,
    
    %Get current indices:
    top_indices = (current_signal_length/2+1) : current_signal_length; 
    bottom_indices = 1 : (current_signal_length/2);
    
    %Get transform:
    for row_counter = 1:current_signal_length
        row = mat_in_transform(row_counter,1:current_signal_length);
        mat_in_transform(row_counter,bottom_indices) = wavelet_downsample_low_pass_periodized(row,QMF);
        mat_in_transform(row_counter,top_indices) = wavelet_downsample_high_pass_periodized(row,QMF);
    end
    for column_counter = 1:current_signal_length
        column = mat_in_transform(1:current_signal_length,column_counter)';
        mat_in_transform(top_indices,column_counter) = wavelet_downsample_high_pass_periodized(column,QMF)';
        mat_in_transform(bottom_indices,column_counter) = wavelet_downsample_low_pass_periodized(column,QMF)';
    end
    current_signal_length = current_signal_length/2;
end
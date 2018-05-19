function mat_reconstructed = wavelet_2D_inverse_transform_orthogonal(mat_in_transform,L_coarsest_scale,QMF)
% IWT2_PO -- Inverse 2-d MRA wavelet transform (periodized, orthogonal)
%  Usage
%    x = IWT2_PO(wc,L,qmf)
%  Inputs
%    wc    2-d wavelet transform [n by n array, n dyadic]
%    L     coarse level
%    qmf   quadrature mirror filter
%  Outputs
%    x     2-d signal reconstructed from wc
%
%  Description
%    If wc is the result of a forward 2d wavelet transform, with
%    wc = FWT2_PO(x,L,qmf), then x = IWT2_PO(wc,L,qmf) reconstructs x
%    exactly if qmf is a nice qmf, e.g. one made by MakeONFilter.
%
%  See Also
%    FWT2_PO, MakeONFilter
%
[signal_length,J_dyadic_length] = quadlength(mat_in_transform);
mat_reconstructed = mat_in_transform;
current_signal_length = 2^(L_coarsest_scale+1);
for j_current_scale = L_coarsest_scale:J_dyadic_length-1
    
    %Get appropriate indices:
    top_indices = (current_signal_length/2+1) : current_signal_length; 
    bottom_indices = 1 : (current_signal_length/2); 
    all_indices = 1 : current_signal_length;
    
    %Get inverse transform by upsampling and combining:
    for column_counter = 1:current_signal_length,
        mat_reconstructed(all_indices,column_counter) =  ...
            wavelet_upsample_low_pass_periodized(mat_reconstructed(bottom_indices,column_counter)',QMF)'  ...
            + wavelet_upsample_high_pass_periodized(mat_reconstructed(top_indices,column_counter)',QMF)';
    end
    for row_counter = 1:current_signal_length,
        mat_reconstructed(row_counter,all_indices) = ...
            wavelet_upsample_low_pass_periodized(mat_reconstructed(row_counter,bottom_indices),QMF)  ...
            + wavelet_upsample_high_pass_periodized(mat_reconstructed(row_counter,top_indices),QMF);
    end
    
    current_signal_length = 2*current_signal_length;
end




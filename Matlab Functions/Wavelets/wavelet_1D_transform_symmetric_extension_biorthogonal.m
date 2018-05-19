function wavelet_coefficients = wavelet_1D_transform_symmetric_extension_biorthogonal(vec_in,L_coarsest_scale,QMF,dual_QMF)
% FWT_SBS -- Forward Wavelet Transform (symmetric extension, biorthogonal, symmetric)
%  Usage
%    wc = FWT_SBS(x,L,qmf,dqmf)
%  Inputs
%    x    1-d signal; arbitrary length
%    L    Coarsest Level of V_0;  L << J
%    qmf    quadrature mirror filter (symmetric)
%    dqmf   quadrature mirror filter (symmetric, dual of qmf)
%  Outputs
%    wc    1-d wavelet transform of x.
%
%  Description
%    1. qmf filter may be obtained from MakePBSFilter
%    2. usually, length(qmf) < 2^(L+1)
%    3. To reconstruct use IWT_SBS
%
%  See Also
%    IWT_SBS, MakePBSFilter
%
%  References
%   Based on the algorithm developed by Christopher Brislawn.
%   See "Classification of Symmetric Wavelet Transforms"
%

[signal_length,J_dyadic_length] = dyadlength(vec_in);

wavelet_coefficients = zeros(1,signal_length);
beta_fine_samples = make_row(vec_in);  % take samples at finest scale as beta-coeffts

dyadic_partition = get_dyadic_partition_of_nondyadic_signals(signal_length);

for j_current_scale = J_dyadic_length-1 : -1 : L_coarsest_scale,
    
    [beta_fine_samples, alpha_fine_samples] = ...
                                wavelet_downsample_symmetric_dual_filter(beta_fine_samples,QMF,dual_QMF);
    
    current_dyad_indices = (dyadic_partition(j_current_scale+1)+1) : dyadic_partition(j_current_scale+2);
    wavelet_coefficients(current_dyad_indices) = alpha_fine_samples;
end
wavelet_coefficients(1:length(beta_fine_samples)) = beta_fine_samples;
wavelet_coefficients = reshape_signal_as_prototype_vec(wavelet_coefficients,vec_in);





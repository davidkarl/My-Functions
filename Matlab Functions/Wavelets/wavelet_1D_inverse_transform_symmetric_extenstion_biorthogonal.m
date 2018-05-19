function x = wavelet_1D_inverse_transform_symmetric_extenstion_biorthogonal(wavelet_coefficients,L_coarsest_scale,QMF,dual_QMF)
% iwt_po -- Inverse Wavelet Transform (symmetric extension, biorthogonal, symmetric)
%  Usage
%    x = IWT_SBS(wc,L,qmf,dqmf)
%  Inputs
%    wc     1-d wavelet transform: length(wc)= 2^J.
%    L      Coarsest scale (2^(-L) = scale of V_0); L << J;
%    qmf     quadrature mirror filter
%    dqmf    dual quadrature mirror filter (symmetric, dual of qmf)
%  Outputs
%    x      1-d signal reconstructed from wc
%  Description
%    Suppose wc = FWT_SBS(x,L,qmf,dqmf) where qmf and dqmf are orthonormal
%    quad. mirror filters made by MakeBioFilter.  Then x can be reconstructed
%    by
%      x = IWT_SBS(wc,L,qmf,dqmf)
%  See Also:
%    FWT_SBS, MakeBioFilter
%

wcoef = make_row(wavelet_coefficients);
[signal_length,J_dyadic_length] = dyadlength(wcoef);

dyadic_partition = get_dyadic_partition_of_nondyadic_signals(signal_length);

x = wcoef(1:dyadic_partition(L_coarsest_scale+1));

for j_current_scale = L_coarsest_scale : J_dyadic_length-1,
    current_scale_indices = (dyadic_partition(j_current_scale+1)+1):dyadic_partition(j_current_scale+2);
    x = wavelet_upsample_symmetric_dual_filter(x, wcoef(current_scale_indices), QMF, dual_QMF);
end
x = reshape_signal_as_prototype_vec(x,wavelet_coefficients);
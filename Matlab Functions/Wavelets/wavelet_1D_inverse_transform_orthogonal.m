function signal_reconstructed = wavelet_1D_inverse_transform_orthogonal(wavelet_coefficients,L_coarsest_scale,QMF)
% IWT_PO -- Inverse Wavelet Transform (periodized, orthogonal)
%  Usage
%    x = IWT_PO(wc,L,qmf)
%  Inputs
%    wc     1-d wavelet transform: length(wc) = 2^J.
%    L      Coarsest scale (2^(-L) = scale of V_0); L << J;
%    qmf    quadrature mirror filter
%  Outputs
%    x      1-d signal reconstructed from wc
%
%  Description
%    Suppose wc = FWT_PO(x,L,qmf) where qmf is an orthonormal quad. mirror
%    filter, e.g. one made by MakeONFilter. Then x can be reconstructed by
%      x = IWT_PO(wc,L,qmf)
%
%  See Also
%    FWT_PO, MakeONFilter
%
wavelet_coefficients = make_row(wavelet_coefficients);
signal_reconstructed = wavelet_coefficients(1:2^L_coarsest_scale);
[signal_length,J_dyadic_length] = dyadlength(wavelet_coefficients);
for j = L_coarsest_scale:J_dyadic_length-1
    %Combine lowpassed and highpassed components:
    signal_reconstructed = wavelet_upsample_low_pass_periodized(signal_reconstructed,QMF) + ...
                           wavelet_upsample_high_pass_periodized( ...
                              wavelet_coefficients(indices_of_wavelet_coefficients_of_given_level(j)) ,QMF);
end
signal_reconstructed = reshape_signal_as_prototype_vec(signal_reconstructed,wavelet_coefficients);









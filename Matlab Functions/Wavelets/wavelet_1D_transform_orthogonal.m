function wavelet_coefficients = wavelet_1D_transform_orthogonal(vec_in,L_coarsest_scale,QMF)
% FWT_PO -- Forward Wavelet Transform (periodized, orthogonal)
%  Usage
%    wc = FWT_PO(x,L,qmf)
%  Inputs
%    x    1-d signal; length(x) = 2^J
%    L    Coarsest Level of V_0;  L << J
%    qmf  quadrature mirror filter (orthonormal)
%  Outputs
%    wc    1-d wavelet transform of x.
%
%  Description
%    1. qmf filter may be obtained from MakeONFilter
%    2. usually, length(qmf) < 2^(L+1)
%    3. To reconstruct use IWT_PO
%
%  See Also
%    IWT_PO, MakeONFilter
%
[signal_length,J_dyadic_length] = dyadlength(vec_in) ;
wavelet_coefficients = zeros(1,signal_length) ;
lowpassed_components = make_row(vec_in);  %take samples at finest scale as beta-coeffts

for j_current_scale = J_dyadic_length-1:-1:L_coarsest_scale
    
    %downsample and get high pass component using the single QMF (which
    %undergoes proper modulation to acquire high pass filter):
    highpassed_components = wavelet_downsample_high_pass_periodized(lowpassed_components,QMF);
    
    %Assign high passed (detail) coefficients to proper indices:
    wavelet_coefficients(indices_of_wavelet_coefficients_of_given_level(j_current_scale)) = highpassed_components;
    
    %Down sample lo
    lowpassed_components = wavelet_downsample_low_pass_periodized(lowpassed_components,QMF) ;
end
wavelet_coefficients(1:(2^L_coarsest_scale)) = lowpassed_components;
wavelet_coefficients = reshape_signal_as_prototype_vec(wavelet_coefficients,vec_in);

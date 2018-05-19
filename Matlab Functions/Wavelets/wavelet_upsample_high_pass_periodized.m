function y = wavelet_upsample_high_pass_periodized(vec_in,QMF)
% UpDyadHi -- Hi-Pass Upsampling operator; periodized
%  Usage
%    u = UpDyadHi(d,f)
%  Inputs
%    d    1-d signal at coarser scale
%    f    filter
%  Outputs
%    u    1-d signal at finer scale
%
%  See Also
%    DownDyadLo, DownDyadHi, UpDyadLo, IWT_PO, aconv
%

y = periodic_convolution_reverse_filter( apply_positive_negative_modulation(QMF), ...
                        circular_shift_right( upsample_operator_insert_zeros_between_samples(vec_in) ) );

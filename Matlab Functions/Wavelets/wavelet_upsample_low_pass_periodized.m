function y = wavelet_upsample_low_pass_periodized(vec_in,QMF)
% UpDyadLo -- Lo-Pass Upsampling operator; periodized
%  Usage
%    u = UpDyadLo(d,f)
%  Inputs
%    d    1-d signal at coarser scale
%    f    filter
%  Outputs
%    u    1-d signal at finer scale
%
%  See Also
%    DownDyadLo, DownDyadHi, UpDyadHi, IWT_PO, iconv
%
y =  periodic_convolution(QMF, upsample_operator_insert_zeros_between_samples(vec_in) );
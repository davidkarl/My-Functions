function y = wavelet_downsample_high_pass_periodized(vec_in,QMF)
% DownDyadHi -- Hi-Pass Downsampling operator (periodized)
%  Usage
%    d = DownDyadHi(x,f)
%  Inputs
%    x    1-d signal at fine scale
%    f    filter
%  Outputs
%    y    1-d signal at coarse scale
%
%  See Also
%    DownDyadLo, UpDyadHi, UpDyadLo, FWT_PO, iconv
%
y = periodic_convolution( apply_positive_negative_modulation(QMF),circular_shift_left(vec_in));
signal_length = length(y);

%take every odd sample:
y = y(1:2:(signal_length-1));
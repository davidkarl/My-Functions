function y = wavelet_downsample_low_pass_periodized(vec_in,QMF)
% DownDyadLo -- Lo-Pass Downsampling operator (periodized)
%  Usage
%    d = DownDyadLo(x,f)
%  Inputs
%    x    1-d signal at fine scale
%    f    filter
%  Outputs
%    y    1-d signal at coarse scale
%
%  See Also
%    DownDyadHi, UpDyadHi, UpDyadLo, FWT_PO, aconv
%
y = periodic_convolution_reverse_filter(QMF,vec_in);
signal_length = length(y);

%take every odd sample:
y = y(1:2:(signal_length-1));
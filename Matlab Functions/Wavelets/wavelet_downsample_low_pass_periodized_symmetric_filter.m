function y = wavelet_downsample_low_pass_periodized_symmetric_filter(vec_in,symmetric_filter)
% DownDyadLo_PBS -- Lo-Pass Downsampling operator (periodized,symmetric)
%  Usage
%    d = DownDyadLo_PBS(x,sf)
%  Inputs
%    x    1-d signal at fine scale
%    sf   symmetric filter
%  Outputs
%    y    1-d signal at coarse scale
%
%  See Also
%    DownDyadHi_PBS, UpDyadHi_PBS, UpDyadLo_PBS, FWT_PBSi, symm_aconv
%
y = periodic_convolution_reverse_symmetric_filter(symmetric_filter,vec_in);

signal_length = length(y);
y = y(1:2:(signal_length-1));
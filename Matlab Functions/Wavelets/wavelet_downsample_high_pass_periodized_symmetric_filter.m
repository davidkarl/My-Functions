function y = wavelet_downsample_high_pass_periodized_symmetric_filter(vec_in,symmetric_filter)
% DownDyadHi_PBS -- Hi-Pass Downsampling operator (periodized,symmetric)
%  Usage
%    d = DownDyadHi_PBS(x,sqmf)
%  Inputs
%    x    1-d signal at fine scale
%    sqmf symmetric filter
%  Outputs
%    y    1-d signal at coarse scale
%
%  See Also
%    DownDyadLo_PBS, UpDyadHi_PBS, UpDyadLo_PBS, FWT_PBS, symm_iconv
%
y = periodic_convolution_symmetric_filter( ...
    apply_positive_negative_modulation_to_mirror_symmetric_filter(symmetric_filter),circular_shift_left(vec_in));

signal_length = length(y);
y = y(1:2:(signal_length-1));




function y_upsampled = wavelet_upsample_low_pass_periodized_symmetric_filter(vec_in,QMF)
% UpDyadLo_PBS -- Lo-Pass Upsampling operator; periodized symmetric
%  Usage
%    u = UpDyadLo_PBS(d,sf)
%  Inputs
%    d    1-d signal at coarser scale
%    sf   symmetric filter
%  Outputs
%    u    1-d signal at finer scale
%
%  See Also
%    DownDyadLo_PBS , DownDyadHi_PBS , UpDyadHi_PBS, IWT_PBS, symm_iconv
%
y_upsampled = ...
    periodic_convolution_symmetric_filter(QMF, upsample_operator_insert_zeros_between_samples(vec_in,2) );

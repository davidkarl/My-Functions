function y_upsampled = wavelet_upsample_high_pass_periodized_symmetric_filter(vec_in,QMF)
% UpDyadHi_PBS -- Hi-Pass Upsampling operator; periodized
%  Usage
%    u = UpDyadHi_PBS(d,f)
%  Inputs
%    d    1-d signal at coarser scale
%    sf   symmetric filter
%  Outputs
%    u    1-d signal at finer scale
%
%  See Also
%    DownDyadLo_PBS, DownDyadHi_PBS, UpDyadLo_PBS, IWT_PBS, symm_aconv
%

y_upsampled = ...
    periodic_convolution_reverse_symmetric_filter( ...
            apply_positive_negative_modulation_to_mirror_symmetric_filter(QMF), ...
                circular_shift_right( upsample_operator_insert_zeros_between_samples(vec_in,2) ) );

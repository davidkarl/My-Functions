function [vec_in_transform] = wavelet_1D_transform_translation_invariant(vec_in,L_coarsest_scale,QMF)
% FWT_TI -- translation invariant forward wavelet transform
%  Usage
%    TIWT = FWT_TI(x,L,qmf)
%  Inputs
%    x        array of dyadic length n=2^J
%    L        degree of coarsest scale
%    qmf      orthonormal quadrature mirror filter
%  Outputs
%    TIWT     stationary wavelet transform table
%             formally same data structure as packet table
%
%  See Also
%    IWT_TI
%

[signal_length,J_dyadic_length] = dyadlength(vec_in);
D_number_of_levels_to_compute = J_dyadic_length - L_coarsest_scale;
vec_in_transform = zeros(signal_length,D_number_of_levels_to_compute+1);
vec_in = make_row(vec_in);
%

vec_in_transform(:,1) = vec_in';
for d_current_depth = 0 : (D_number_of_levels_to_compute-1),
    for b_block_index = 0:(2^d_current_depth-1),
        %get current samples according to depth, signal length, and block:
        current_samples = ...
            vec_in_transform(wavelet_packet_table_indexing(d_current_depth,b_block_index,signal_length),1)';
        
        %Decompose and downsample current samples using high pass and low pass filters 
        %CREATED FROM THE SAME QMF:
        high_passed_downsampled_right = wavelet_downsample_high_pass_periodized(current_samples,QMF);
        high_passed_downsampled_left = wavelet_downsample_high_pass_periodized(circular_shift_right(current_samples),QMF);
        low_passed_downsampled_right = wavelet_downsample_low_pass_periodized(current_samples,QMF);
        low_passed_downsampled_left = wavelet_downsample_low_pass_periodized(circular_shift_right(current_samples),QMF);
        
        %Insert appropriate downsampled samples into the transform appropriate indices:
        vec_in_transform(wavelet_packet_table_indexing(d_current_depth+1, 2*b_block_index ,signal_length),...
                                                        d_current_depth+2) = high_passed_downsampled_right';
        vec_in_transform(wavelet_packet_table_indexing(d_current_depth+1, 2*b_block_index+1, signal_length),...
                                                        d_current_depth+2) = high_passed_downsampled_left';
        vec_in_transform(wavelet_packet_table_indexing(d_current_depth+1, 2*b_block_index ,signal_length), ...
                                                                        1 ) = low_passed_downsampled_right';
        vec_in_transform(wavelet_packet_table_indexing(d_current_depth+1, 2*b_block_index+1 ,signal_length), ...
                                                                        1 ) = low_passed_downsampled_left';
    end
end







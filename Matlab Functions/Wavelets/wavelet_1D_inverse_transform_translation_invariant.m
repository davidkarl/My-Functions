function x = wavelet_1D_inverse_transform_translation_invariant(...
                                                translation_invariant_wavelet_transform_table,...
                                                quadrature_mirror_filter)
% IWT_TI -- Invert translation invariant wavelet transform
%  Usage
%    x = IWT_TI(TIWT,qmf)
%  Inputs
%    TIWT     translation-invariant wavelet transform table
%    qmf      quadrature mirror filter
%  Outputs
%    x        1-d signal reconstructed from translation-invariant
%             transform TIWT
%
%  See Also
%    FWT_TI
%
[signal_length,D_number_of_levels_plus_one] = size(translation_invariant_wavelet_transform_table);
D_number_of_levels = D_number_of_levels_plus_one-1;
J_number_of_possible_levels = log2(signal_length);
L = J_number_of_possible_levels-D_number_of_levels;
%
vec_in_transform = translation_invariant_wavelet_transform_table;
%
signal_parts = vec_in_transform(:,1)';
for d_current_depth = D_number_of_levels-1:-1:0
    for b_block_index = 0:(2^d_current_depth-1)
        
        %get appropriate parts from the wavelet transform:
        high_passed_downsampled_right = vec_in_transform( wavelet_packet_table_indexing(d_current_depth+1,2*b_block_index,signal_length),...
                                                d_current_depth+2)';
        high_passed_downsampled_left = vec_in_transform(wavelet_packet_table_indexing(d_current_depth+1,2*b_block_index+1,signal_length),...
                                                d_current_depth+2)';
        low_passed_downsampled_right = signal_parts(wavelet_packet_table_indexing(d_current_depth+1,2*b_block_index,signal_length) );
        low_passed_downsampled_left = signal_parts(wavelet_packet_table_indexing(d_current_depth+1,2*b_block_index+1,signal_length) );
        
        %upsample appropriate terms from the low and high pass left and right terms:
        low_pass_terms = (...
            wavelet_upsample_low_pass_periodized(low_passed_downsampled_right,quadrature_mirror_filter) ...
          + circular_shift_left( wavelet_upsample_low_pass_periodized(low_passed_downsampled_left,quadrature_mirror_filter)))/2;
        
        high_pass_terms = (...
            wavelet_upsample_high_pass_periodized(high_passed_downsampled_right,quadrature_mirror_filter) ...
          + circular_shift_left( wavelet_upsample_high_pass_periodized(high_passed_downsampled_left,quadrature_mirror_filter)))/2;
        
        %join the upsampled terms:
        signal_parts(packet(d_current_depth,b_block_index,signal_length)) = ...
                                                                        low_pass_terms + high_pass_terms;
    end
end
x = signal_parts;







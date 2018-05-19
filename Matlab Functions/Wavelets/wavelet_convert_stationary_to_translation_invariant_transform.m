function [translation_invariant_transform_table] = ...
               wavelet_convert_stationary_to_translation_invariant_transform(stationary_wavelet_transform_table)
% Stat2TI -- Convert Stationary Wavelet Transform to Translation-Invariant Transform
%  Usage
%    TIWT = Stat2TI(StatWT)
%  Inputs
%    StatWT  stationary wavelet transform table as FWT_Stat
%  Outputs
%    TIWT    translation-invariant transform table as FWT_TI
%
%  See Also
%    Stat2TI, FWT_TI, FWT_Stat
%

translation_invariant_transform_table = stationary_wavelet_transform_table;
[signal_length,D1] = size(stationary_wavelet_transform_table);
number_of_levels = D1-1;
index_vec = 1;

for depth_counter = 1:number_of_levels,
    number_of_blocks_in_current_depth = 2^depth_counter;
    nk = signal_length/number_of_blocks_in_current_depth;

    index_vec = [ (index_vec+number_of_blocks_in_current_depth/2); index_vec];
    index_vec = index_vec(:)';

    for block_counter = 0:(number_of_blocks_in_current_depth-1),
        wavelet_packet_table_index = wavelet_packet_table_indexing(depth_counter,block_counter,signal_length);
        translation_invariant_transform_table(depth_counter*signal_length + wavelet_packet_table_index) ...
                = stationary_wavelet_transform_table(depth_counter*signal_length + (index_vec(block_counter+1):number_of_blocks_in_current_depth:signal_length));
    end
end

for block_counter = 0:(number_of_blocks_in_current_depth-1),
    wavelet_packet_table_index = wavelet_packet_table_indexing(depth_counter,block_counter,signal_length);
    translation_invariant_transform_table(wavelet_packet_table_index) ...
            = stationary_wavelet_transform_table((index_vec(block_counter+1):number_of_blocks_in_current_depth:signal_length));
end
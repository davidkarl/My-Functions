function [final_signal] = smooth_vec_by_decay(input_signal,decay_per_block_dB,number_of_blocks)

final_signal = abs(input_signal);

for k=1:length(input_signal)
    center_value = input_signal(k);
    current_indices = max(1,k-number_of_blocks) : 1 : min(k+number_of_blocks,length(input_signal));
    current_centered_indices = current_indices - k;
    current_values_updates = center_value * 10.^(-decay_per_block_dB*abs(current_centered_indices)/20);
    final_signal(current_indices) = max(current_values_updates(:),final_signal(current_indices));
end




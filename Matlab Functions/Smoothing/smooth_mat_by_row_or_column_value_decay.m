function [input_mat] = smooth_mat_by_row_or_column_value_decay(...
    input_mat,...
    attack_decay_per_block,...
    release_decay_per_block,...
    gains_history_full_flipped_indices,...
    center_of_search_window_in_gains_history,...
    center_of_search_window_index,...
    noise_attenuation_factor,...
    channel_counter)
 

gains_history_above_noise = input_mat(:,:,channel_counter);
gains_history_number_of_blocks = size(input_mat,2);

noise_attenuation_constants = noise_attenuation_factor * ones(size(input_mat,1),1);
%(*) if the gain before or after is lower then what one gets when we decay the current center gain
% then RAISE it according to decay rate from center gain:
%HOLD (backward in time):
for history_frame_counter = center_of_search_window_in_gains_history-1 : -1 : 1
    current_index = gains_history_full_flipped_indices(history_frame_counter+1,channel_counter);
    next_index = gains_history_full_flipped_indices(history_frame_counter,channel_counter);
    gains_history_above_noise(:,next_index) = max( gains_history_above_noise(:,current_index)*attack_decay_per_block, max(noise_attenuation_constants,gains_history_above_noise(:,next_index)) );
end
%RELEASE (forward in time):
for history_frame_counter = center_of_search_window_in_gains_history+1 : +1 : gains_history_number_of_blocks
    current_index = gains_history_full_flipped_indices(history_frame_counter-1,channel_counter);
    next_index = gains_history_full_flipped_indices(history_frame_counter,channel_counter);
    gains_history_above_noise(:,next_index) = max( gains_history_above_noise(:,current_index)*release_decay_per_block, max(noise_attenuation_constants,gains_history_above_noise(:,next_index)) );
end


input_mat(:,:,channel_counter) = gains_history_above_noise;





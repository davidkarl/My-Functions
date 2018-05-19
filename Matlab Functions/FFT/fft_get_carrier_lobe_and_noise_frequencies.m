function [carrier_lobe_start,carrier_lobe_stop,noise_floor_left_start,noise_floor_left_stop,noise_floor_right_start,noise_floor_right_stop] = ...
    fft_get_carrier_lobe_and_noise_frequencies(average_max_frequency,effective_lobe_BW_one_sided,BW)


carrier_lobe_start = average_max_frequency - effective_lobe_BW_one_sided;
carrier_lobe_stop = average_max_frequency + effective_lobe_BW_one_sided;
noise_floor_left_start = average_max_frequency-BW/2;
noise_floor_left_stop = average_max_frequency-effective_lobe_BW_one_sided;
noise_floor_right_start = average_max_frequency+effective_lobe_BW_one_sided;
noise_floor_right_stop = average_max_frequency+BW/2;



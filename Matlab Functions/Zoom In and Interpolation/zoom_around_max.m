function [zoomed_signal,start_index,stop_index] = zoom_around_max(signal,max_radius_around_max)

[max_value,max_index] = max(signal);
start_index = max(1,max_index-max_radius_around_max);
stop_index = min(length(signal),max_index+max_radius_around_max);
zoomed_signal = signal(start_index:stop_index);

 
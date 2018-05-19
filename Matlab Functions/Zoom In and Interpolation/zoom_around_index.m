function [zoomed_signal,start_index,stop_index] = zoom_around_index(signal,index, max_radius_around_index)

start_index = max(1,index-max_radius_around_index);
stop_index = min(length(signal),index+max_radius_around_index);
zoomed_signal = signal(start_index:stop_index);

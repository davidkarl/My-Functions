function [zoomed_signal,start_index,stop_index] = zoom_around_center(signal, max_radius_around_center)

center_index = ceil(length(signal)/2)+(1-mod(length(signal),2));
start_index = max(1,center_index-max_radius_around_center);
stop_index = min(length(signal),center_index+max_radius_around_center);
zoomed_signal = signal(start_index:stop_index);

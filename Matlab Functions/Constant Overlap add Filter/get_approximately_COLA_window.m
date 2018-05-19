function [frame_window] = get_approximately_COLA_window(samples_per_frame,overlap_samples_per_frame)

non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame;

%build approximately COLA window:
if overlap_samples_per_frame>1 && overlap_samples_per_frame<floor(samples_per_frame/2)
    hanning_edge = make_column(hann(2*overlap_samples_per_frame-1,'symmetric'));
    part_one = hanning_edge(1:overlap_samples_per_frame);
    part_two = ones(samples_per_frame-2*overlap_samples_per_frame,1);
    part_three = hanning_edge(end-overlap_samples_per_frame+1:end);
    frame_window = [part_one;part_two;part_three];
else
    hanning_edge = make_column(hann(2*overlap_samples_per_frame-1,'symmetric'));
    part_one = hanning_edge(end-overlap_samples_per_frame+1:end);
    part_one = [ones(non_overlapping_samples_per_frame,1) ; part_one];
    
    part_two = hanning_edge(1:overlap_samples_per_frame);
    part_two = [part_two ; ones(non_overlapping_samples_per_frame,1)];
    
    frame_window = (part_one.*part_two).^(1/2);
end

%sample sum of frames in one place (instead of calculation the sum of all
%the window samples and taking it's max) to get normalization factor:
sum_for_normalization = 0;
for k = 1:floor(samples_per_frame/non_overlapping_samples_per_frame)
    sum_for_normalization = sum_for_normalization + frame_window(1+(k-1)*non_overlapping_samples_per_frame);
end

%normalize window so that overlapping all frames will give approximately perfect reconstruction:
frame_window = frame_window ./ sum_for_normalization;



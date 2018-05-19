% function [wsum]=add_window(win,L,R,ncopy)
%
% function to overlap-add ncopy copies of window, each shifted by R samples,
% and plot results

% Inputs:
%   win: window sequence to be overlap-added
%   L: window length
%   R: window shift
%   ncopy: number of copies of overlap-added window to be added
%
% Output:
%   wsum: window sum
     
clear all;
L=2048;
R=2000;
win = hanning(L)';

for bla=1:9
    
samples_per_frame = 2048;
overlap_samples_per_frame = 1513;
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame;
if overlap_samples_per_frame>1 && overlap_samples_per_frame<floor(samples_per_frame/2)
    hanning_edge = make_column(hann(2*overlap_samples_per_frame-1,'symmetric'));
    part_one = hanning_edge(1:overlap_samples_per_frame);
    part_two = ones(samples_per_frame-2*overlap_samples_per_frame,1);
    part_three = hanning_edge(end-overlap_samples_per_frame+1:end);
    frame_window = [part_one;part_two;part_three];
else
    hanning_edge = make_column(hann(2*overlap_samples_per_frame-1,'symmetric'));
    part_one = hanning_edge(end-overlap_samples_per_frame+1:end);
    part_one = [ones(non_overlapping_samples_per_frame,1);part_one];
    part_two = hanning_edge(1:overlap_samples_per_frame);
    part_two = [hanning_edge(1:overlap_samples_per_frame);ones(non_overlapping_samples_per_frame,1)];
    frame_window = part_one.*part_two;
    frame_window = frame_window.^1/2;
end
L = samples_per_frame;
R = non_overlapping_samples_per_frame;
win = make_row(frame_window);

ncopy = ceil(L/R);
list = hsv(ncopy);
wsum=0;
blabla=0; 
% plot all partial sums
    bla = L+(ncopy-1)*R;  
    wsum(1:L+(ncopy-1)*R)=0;
%     figure
    plot(0:length(wsum)-1,wsum,'k:');hold on;
    istart=1; 
    for iadd=1:ncopy
        iend = istart+L-1;
        wsum(istart:iend)=wsum(istart:istart+L-1)+win(1:L);
        blabla = blabla + win(1+(iadd-1)*R);
        plot(0:length(wsum)-1,wsum,'LineWidth',2,'color',list(iadd,:));
        istart=istart+R;
    end
    xlabel('samples');ylabel('partial sums');grid on; 
end
1;        
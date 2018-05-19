function [smoothed_signal] = conv_without_end_effects(input_signal,smoothing_window,flag_use_fft)
%(1). assuming window length is odd!!!
%(2). one doesn't have to normalize the window, it's done here automatically

if flag_use_fft==1
    smoothed_signal = conv_fft(input_signal,smoothing_window)/sum(smoothing_window);
else
    smoothed_signal = conv(input_signal,smoothing_window,'same')/sum(smoothing_window);
end

for k=1:floor(length(smoothing_window)/2)   
    smoothed_signal(k) = sum(input_signal(k:length(smoothing_window)-1) .* smoothing_window(k+1:end)) / sum(smoothing_window(k+1:end));
    smoothed_signal(end-k+1) = sum(input_signal(end-length(smoothing_window)+1+k : end) .* smoothing_window(1:end-k)) / sum(smoothing_window(1:end-k));
end


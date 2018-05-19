%test filtering with overlap

%get input signal:
clear all;
Fs = 44100;
Fc = 8000;
t_vec = my_linspace(0,1/Fs,Fs*20);
input_signal = 0.05 * sin(2*pi*Fc*t_vec);

[input_signal,bla] = wavread('C:\Users\master\Desktop\matlab\parameter experiment sound files\counting forward.wav');
input_signal = input_signal(:,1);
input_signal = input_signal(5000:end);
input_signal = 0.01 * input_signal;

%get parameters:
samples_per_frame = 2048;
overlap_samples_per_frame = 2000;
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

% frame_window = hanning(samples_per_frame,'periodic');
% frame_window = ones(samples_per_frame,1);
 
%get filter:
start_frequency = 300;
stop_frequency = 20000;
filter_length1 = 600;
signal_filter = get_filter_1D('hann',20,filter_length1,Fs,start_frequency,stop_frequency,'bandpass');
% group_delay1 = round(mean(grpdelay(signal_filter)));
group_delay1 = floor(filter_length1/2)+1;
FFT_size = 2^nextpow2(samples_per_frame+filter_length1-1);
signal_filter_fft = make_column(fft(signal_filter.Numerator,FFT_size));

start_frequency = 300;
stop_frequency = 20000;
filter_length2 = 1000;
signal_filter2 = get_filter_1D('hann',20,filter_length2,Fs,start_frequency,stop_frequency,'bandpass');
% group_delay2 = round(mean(grpdelay(signal_filter2)));
group_delay2 = floor(filter_length2/2)+1;
FFT_size = 2^nextpow2(samples_per_frame+filter_length2-1);
signal_filter_fft2 = make_column(fft(signal_filter2.Numerator,FFT_size));

%buffer signal:
no_overlap_signal_matrix = buffer(input_signal,samples_per_frame,0,'nodelay');
input_signal_matrix = buffer(input_signal,samples_per_frame,overlap_samples_per_frame,'nodelay');
previous_overlap = zeros(overlap_samples_per_frame,1);
% previous_overlap = zeros(2*group_delay+1,1);

%audio player object:
audio_player_object = dsp.AudioPlayer;
audio_player_object.SampleRate = Fs;


%Loop over and sound resulting signal:
final_signal_to_sound1 = [];
final_signal_to_sound2 = [];
final_signal1 = zeros(length(input_signal),1);
final_signal2 = zeros(length(input_signal),1);
for k=1:size(input_signal_matrix,2)
   
    %get current frame:
    current_frame = input_signal_matrix(:,k);
    current_frame2 = input_signal_matrix(:,k);
%     current_frame = no_overlap_signal_matrix(:,k);
    
    %window frame:
    current_frame_windowed = current_frame .* frame_window;
    
    %fft windowed frame:
    current_frame_fft = fft(current_frame_windowed , FFT_size);
    
    %multiply by filter fft:
    current_frame_fft_filtered1 = current_frame_fft .* signal_filter_fft;
    current_frame_fft_filtered2 = current_frame_fft .* signal_filter_fft2;
    
    %get filtered time domain signal by ifft:
    current_frame_filtered_expanded1 = real(ifft(current_frame_fft_filtered1));
    current_frame_filtered_expanded2 = real(ifft(current_frame_fft_filtered2)); 
    
%     %get only valid part of convolution:
%     current_frame_filtered_valid = current_frame_filtered_expanded(group_delay+1:group_delay+samples_per_frame);
%     current_signal_to_sound = current_frame_filtered_valid(1:end);
%     current_signal_to_sound(1:overlap_samples_per_frame) = current_signal_to_sound(1:overlap_samples_per_frame) + previous_overlap;
%     previous_overlap = current_frame_filtered_valid( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame );
%     final_signal2 = [final_signal2;current_signal_to_sound];
    
% %     current_frame_filtered_valid = current_frame_filtered_expanded(group_delay+1:group_delay+samples_per_frame+group_delay);
%     current_frame_filtered_valid = current_frame_filtered_expanded(1:end);
%     current_signal_to_sound = current_frame_filtered_valid;
%     current_signal_to_sound(1:2*group_delay+1) = current_signal_to_sound(1:2*group_delay+1) + previous_overlap;
%     previous_overlap = current_frame_filtered_expanded(group_delay+1+samples_per_frame-group_delay : group_delay+1+samples_per_frame+group_delay);
%     current_signal_to_sound = current_signal_to_sound(1:samples_per_frame);


    %use overlap add:
    filter_length_difference = filter_length2 - filter_length1;
    filter_group_delay_difference = group_delay2 - group_delay1;
    
    final_signal1((k-1)*non_overlapping_samples_per_frame+1 : (k-1)*non_overlapping_samples_per_frame + FFT_size) = ...
        final_signal1((k-1)*non_overlapping_samples_per_frame+1   : (k-1)*non_overlapping_samples_per_frame + FFT_size) + current_frame_filtered_expanded1;
    
    final_signal2((k-1)*non_overlapping_samples_per_frame+1 : (k-1)*non_overlapping_samples_per_frame + FFT_size - filter_group_delay_difference) = ...
        final_signal2((k-1)*non_overlapping_samples_per_frame+1 : (k-1)*non_overlapping_samples_per_frame + FFT_size - filter_group_delay_difference) + current_frame_filtered_expanded2(filter_group_delay_difference+1:end);
    
    
    current_signal_valid1 = final_signal1((k-1)*non_overlapping_samples_per_frame+1 : (k)*non_overlapping_samples_per_frame);
    current_signal_valid2 = final_signal2((k-1)*non_overlapping_samples_per_frame+1 : (k)*non_overlapping_samples_per_frame);
    final_signal_to_sound1 = [final_signal_to_sound1;current_signal_valid1];
    final_signal_to_sound2 = [final_signal_to_sound2;current_signal_valid2];


%     step(audio_player_object,no_overlap_signal_matrix(:,k));
%     step(audio_player_object,current_signal_to_sound);
end

% sound(final_signal1,44100);
% sound(final_signal_to_sound2,44100);
final_signal_to_sound1 = final_signal_to_sound1(1:length(final_signal_to_sound2));
sound(final_signal_to_sound1+final_signal_to_sound2,44100);

% figure
% plot(current_frame_filtered_expanded);
% clear sound;





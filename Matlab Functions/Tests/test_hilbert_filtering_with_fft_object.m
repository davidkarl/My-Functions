%test hilbert filtering

%get input signal:
Fs = 44100;
Fc = 12000;
t_vec = make_column(my_linspace(0,1/Fs,Fs*20));

[rodena,bla] = wavread('C:\Users\master\Desktop\matlab\parameter experiment sound files\counting forward.wav');
rodena = rodena(:,1);
rodena = rodena(1:end);
Fc2 = 2000;
input_signal = 0.5 * sin(2*pi*Fc*t_vec(1:length(rodena))+0.1*rodena);
% input_signal = 0.5 * sin(2*pi*Fc*t_vec + 0.001*sin(2*pi*Fc2*t_vec));

%get parameters:
samples_per_frame = 2048;
overlap_samples_per_frame = 1024;
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
    part_two = [part_two;ones(non_overlapping_samples_per_frame,1)];
    frame_window = part_one.*part_two;
    frame_window = frame_window.^1/2;
end



%get signal filter:
start_frequency = 4000;
stop_frequency = 18000;
carrier_filter_length = 300;
carrier_filter = get_filter_1D('hamming',20,carrier_filter_length,Fs,start_frequency,stop_frequency,'bandpass');
carrier_filter_group_delay = floor(carrier_filter_length/2)+1;
%get hilbert filter:
hilbert_filter_length = 200;
hilbert_filter_design = fdesign.hilbert('N,Tw',hilbert_filter_length,0.02*(Fs/2),Fs);
hilbert_filter = design(hilbert_filter_design,'FIR');
hilbert_filter.PersistentMemory = true;
hilbert_filter.States = 0;
hilbert_filter_group_delay = round(mean(grpdelay(hilbert_filter)));
%Cascade signal+equalizer filter:
carrier_plus_hilbert_filter = dfilt.cascade(hilbert_filter,carrier_filter);
carrier_plus_hilbert_filter = dfilt.dffir(carrier_plus_hilbert_filter.impz);
carrier_plus_hilbert_filter.PersistentMemory = true; 
carrier_plus_hilbert_filter.States = 0;
carrier_plus_hilbert_filter_length = length(carrier_plus_hilbert_filter.impz);
% carrier_plus_hilbert_filter_group_delay = round(mean(grpdelay(carrier_plus_hilbert_filter)));
carrier_plus_hilbert_filter_group_delay = carrier_filter_group_delay + hilbert_filter_group_delay;
FFT_size_carrier_plus_hilbert_filter = 2^nextpow2(samples_per_frame+(carrier_plus_hilbert_filter_length-1));
%create filter fft:
FFT_size = 2^nextpow2(samples_per_frame+carrier_filter_length+hilbert_filter_length-1);
carrier_plus_hilbert_filter_fft = (fft(carrier_plus_hilbert_filter.impz,FFT_size_carrier_plus_hilbert_filter));
hilbert_filter_fft = fft(hilbert_filter.impz,FFT_size_carrier_plus_hilbert_filter);
carrier_filter_fft = make_column(fft(carrier_filter.Numerator,FFT_size));

%buffer signal:
bla = 4*overlap_samples_per_frame;
no_overlap_signal_matrix = buffer(input_signal,overlap_samples_per_frame,0,'nodelay');
input_signal_matrix = buffer(input_signal,samples_per_frame,overlap_samples_per_frame,'nodelay');
previous_overlap = zeros(overlap_samples_per_frame,1);
% previous_overlap = zeros(2*group_delay+1,1);

%audio player object:
audio_player_object = dsp.AudioPlayer;
audio_player_object.SampleRate = Fs;

previous = 0;
phase_difference = zeros(non_overlapping_samples_per_frame,1);
previous_real = zeros(overlap_samples_per_frame,1);
previous_imag = zeros(overlap_samples_per_frame,1);
analytic_signal_object = dsp.AnalyticSignal(200);
t_vec = make_column(my_linspace(0,1/Fs,non_overlapping_samples_per_frame));
% t_vec = zeros(overlap_samples_per_frame,1);

lookahead_tracker_real=zeros(FFT_size,1);
lookahead_tracker_imag=zeros(FFT_size,1);
 
%create fft_overlap_add_objects:
number_of_samples_to_skip_at_ifft_output = floor(carrier_plus_hilbert_filter_length/2) - floor(carrier_filter_length/2);
[filter_object_imag] = fft_overlap_add_object(carrier_plus_hilbert_filter,Fs,samples_per_frame,overlap_samples_per_frame,2,number_of_samples_to_skip_at_ifft_output);
[filter_object_real] = fft_overlap_add_object(carrier_filter,Fs,samples_per_frame,overlap_samples_per_frame,2,0);
filter_object_real.equate_latency(filter_object_imag);

%Loop over and sound resulting signal:
for k=1:size(input_signal_matrix,2)  
    
    %get current frame:
    current_frame = input_signal_matrix(:,k);
%     current_frame = no_overlap_signal_matrix(:,k);
    
    %window frame:
    current_frame_windowed = current_frame .* frame_window;
%     current_frame_windowed = current_frame .* 1;
    
    %fft windowed frame:
    current_frame_fft = fft(current_frame_windowed , FFT_size);
    
    %filter frames:
    current_valid_real = filter_object_real.filter(current_frame_fft,2);
    current_valid_imag = filter_object_imag.filter(current_frame_fft,2);
    analytic_signal = current_valid_real + 1i*current_valid_imag;
    
%     %multiply by filter fft:
%     current_frame_fft_carrier_filtered = current_frame_fft .* carrier_filter_fft;
%     current_frame_fft_hilbert_filtered = current_frame_fft .* hilbert_filter_fft;
%     current_frame_fft_carrier_plus_hilber_filtered = current_frame_fft .* carrier_plus_hilbert_filter_fft;
%     
%     %get filtered time domain signal by ifft:
%     current_frame_carrier_filtered_ifft = real(ifft(current_frame_fft_carrier_filtered));
%     current_frame_hilbert_filtered_ifft = real(ifft(current_frame_fft_hilbert_filtered));
%     current_frame_carrier_plus_hilbert_filtered_ifft = real(ifft(current_frame_fft_carrier_plus_hilber_filtered));
% 
%     
%     %get only valid part of convolution:
%     current_frame_hilbert = current_frame_hilbert_filtered_ifft(hilbert_filter_group_delay+1:hilbert_filter_group_delay+samples_per_frame);
%     current_frame_filtered_real = current_frame_carrier_filtered_ifft(carrier_filter_group_delay+1:carrier_filter_group_delay+samples_per_frame);
%     current_frame_filtered_imag = current_frame_carrier_plus_hilbert_filtered_ifft(carrier_plus_hilbert_filter_group_delay+1:carrier_plus_hilbert_filter_group_delay+samples_per_frame);
%     
%     filter_length_difference = carrier_plus_hilbert_filter_length - carrier_filter_length;
%     group_delay_difference = floor(carrier_plus_hilbert_filter_length/2) - floor(carrier_filter_length/2);
%     
%     lookahead_tracker_real = lookahead_tracker_real + current_frame_carrier_filtered_ifft;
%     current_valid_real = lookahead_tracker_real(1 : non_overlapping_samples_per_frame);
%     lookahead_tracker_real = [lookahead_tracker_real(non_overlapping_samples_per_frame+1:end) ; zeros(non_overlapping_samples_per_frame,1)];
%      
%     lookahead_tracker_imag(1:end - group_delay_difference) = lookahead_tracker_imag(1:end - group_delay_difference) + current_frame_carrier_plus_hilbert_filtered_ifft(group_delay_difference+1:end);
%     current_valid_imag = lookahead_tracker_imag(1 : non_overlapping_samples_per_frame);
%     lookahead_tracker_imag = [lookahead_tracker_imag(non_overlapping_samples_per_frame+1:end) ; zeros(non_overlapping_samples_per_frame,1)];
%     analytic_signal = current_valid_real + 1i*current_valid_imag;
    

    t_vec = (k-1)*non_overlapping_samples_per_frame*1/Fs + make_column(my_linspace(1/Fs,1/Fs,non_overlapping_samples_per_frame));
    analytic_signal = analytic_signal.*exp(-1i*2*pi*Fc*t_vec);
    phase_difference(2:end) = angle(analytic_signal(2:end).*conj(analytic_signal(1:end-1)));
    first = analytic_signal(1);
    phase_difference(1) = angle(first.*conj(previous));
    previous = analytic_signal(end);
%     phase_difference = phase_difference - mean(phase_difference);
    

    %sound:
%     step(audio_player_object,no_overlap_signal_matrix(:,k));
    step(audio_player_object,1*phase_difference);
end
 











% figure
% plot(current_frame_filtered_expanded);






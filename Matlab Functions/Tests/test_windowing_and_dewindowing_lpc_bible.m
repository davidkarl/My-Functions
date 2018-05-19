%test windowing and dewindowing from lpc bible:

%read input signal:
[input_signal,Fs] = wavread('shirt_2mm_ver_200m_audioFM final demodulated audio150-3000[Hz]');

%initialize framing parameters:
samples_per_frame = 2048;
overlapping_samples_per_frame = (samples_per_frame * 1/2);	
non_overlapping_samples_per_frame = samples_per_frame - overlapping_samples_per_frame ; %=segment step
FFT_size = 2^(nextpow2(samples_per_frame));
total_number_of_frames = floor(length(input_signal)/non_overlapping_samples_per_frame)-1;

%Initialize windows:
hamming_window = make_column(hamming(samples_per_frame));
hanning_window_for_frame_edge_averaging = make_column(hanning(2*overlapping_samples_per_frame-1));
deframing_window = [hanning_window_for_frame_edge_averaging(1:overlapping_samples_per_frame);ones(samples_per_frame-2*overlapping_samples_per_frame,1);...
    hanning_window_for_frame_edge_averaging(overlapping_samples_per_frame:end)]./hamming_window;

%initialize signal source and buffer objects:
signal_source_object = dsp.SignalSource;
signal_source_object.Signal = input_signal;
signal_source_object.SamplesPerFrame = non_overlapping_samples_per_frame;
signal_buffer_object = dsp.Buffer;
signal_buffer_object.Length = samples_per_frame;
signal_buffer_object.OverlapLength = overlapping_samples_per_frame;

%initialize final signal:
final_signal = zeros(length(input_signal),1);

%Loop over input signal frames:
for frame_counter=1:total_number_of_frames
   %get current indices:
   start_index = (frame_counter-1)*non_overlapping_samples_per_frame+1;
   stop_index = (frame_counter)*non_overlapping_samples_per_frame+overlapping_samples_per_frame;
    
   %get current buffered frame:
   current_frame = step(signal_buffer_object,step(signal_source_object));
   
   %frame current frame by window:
   current_frame = current_frame .* hamming_window; 
   
   %deframe current frame by window:
   current_frame_deframed = current_frame .* deframing_window;
   
   %put final result into final signal buffer:
   final_signal( start_index:stop_index) = final_signal(start_index:stop_index) + current_frame_deframed;
end 


1;


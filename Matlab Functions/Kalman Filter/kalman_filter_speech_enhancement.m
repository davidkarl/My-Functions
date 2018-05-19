function output_signal=kalman_filter_speech_enhancement(input_signal)
% This function performs the total processing of the input vector
% using Kalman filtering
% It is based on procsig2 and getpns2
% Usage: yf=procsig3(x)

% Set some constants first
samples_per_frame=256   ; % window length, should be power of two
overlap_factor=2  ; % overlap factor
non_overlapping_samples_per_frame = samples_per_frame/overlap_factor ; % window shift

AR_model_order_speech=16 ;   % speech model order
AR_model_order_noise=16 ;   % noise model order

minimum_bias_factor=1.5 ; % estimate bias factor

% lnga=7 ; 
% lngb=8 ;
segment_size=10 ; 
circular_buffer_size=10 ;
total_size_where_minimum_is_searched = segment_size*circular_buffer_size ; % length of the window for minimum evaluation
raw_power_spectrum_smoothing_factor=0.9 ;     % smoothing factor 



input_signal=input_signal(:)' ;
total_number_of_samples=length(input_signal) ;

number_of_frames=floor((total_number_of_samples-samples_per_frame)/non_overlapping_samples_per_frame) ; % how many windows
hanning_window=hanning(samples_per_frame) ;       % window weights
hanning_window=hanning_window * (samples_per_frame/overlap_factor/sum(hanning_window)) ;  % adjust to proper scaling


input_signal=input_signal-sum(input_signal)/total_number_of_samples ;       % remove the mean
smoothed_raw_power_spectrum = zeros(1,samples_per_frame/2) ;   % smoothed power spectrum
circular_buffer = zeros(circular_buffer_size,samples_per_frame/2) ; % circular buffer of sps history
circular_buffer_counter = 1 ;              % index to circular buffer
circular_buffer_minimum = zeros(1,samples_per_frame/2) ; % current minimum of the circular buffer
last_segment_minimum = zeros(1,samples_per_frame/2) ; % current minimum of the last segment
last_segment_counter = segment_size ;           % length of the last segment
noise_power_spectrum_estimate = zeros(1,samples_per_frame/2) ;   % noise estimate power spectrum
output_signal = zeros(1,total_number_of_samples) ;      % output

sn=zeros(number_of_frames+1,samples_per_frame/2) ;
for i=0:number_of_frames
    
    %get start and stop indices:
    start_index=1+i*non_overlapping_samples_per_frame ;
    stop_index=samples_per_frame+i*non_overlapping_samples_per_frame ; 
    
    %get current frame, current frame fft, spectrum, and smoothed spectrum:
    current_frame = input_signal(start_index:stop_index) ;      
    current_frame_fft = fft(current_frame) ;
    current_frame_fft = current_frame_fft(1:samples_per_frame/2); 
    current_frame_power_spectrum = current_frame_fft.*conj(current_frame_fft) ;   
    smoothed_raw_power_spectrum = raw_power_spectrum_smoothing_factor*smoothed_raw_power_spectrum ...
        + (1-raw_power_spectrum_smoothing_factor)*current_frame_power_spectrum ;
    
    %Track minimum:
    if last_segment_counter>=segment_size
        circular_buffer(circular_buffer_counter,:)=last_segment_minimum ;   
        circular_buffer_counter=mod(circular_buffer_counter,circular_buffer_size)+1 ; 
        circular_buffer_minimum=min(circular_buffer) ;
        last_segment_minimum=smoothed_raw_power_spectrum ;
        last_segment_counter=1 ;
    else
        last_segment_minimum=min(last_segment_minimum,smoothed_raw_power_spectrum) ;
        last_segment_counter=last_segment_counter+1 ;
    end
    
    %Get noise power spectrum using bias correction:
    noise_power_spectrum_estimate = minimum_bias_factor*min(last_segment_minimum,circular_buffer_minimum) ;       % new noise estimate
    
    % smoothed speech power spectrum estimate:
    smoothed_speech_power_spectrum_estimate = smoothed_raw_power_spectrum-noise_power_spectrum_estimate ;
    
    % now we have the speech power spectrum in sss
    % and the noise power spectrum in nps
    [AR_paraemters_speech,AR_gain_speech,auto_correlation_sequence_speech] = spectrum_to_AR_parameters(smoothed_speech_power_spectrum_estimate,AR_model_order_speech);
    [AR_parameters_noise,AR_gain_noise,auto_correlation_sequence_noise] = spectrum_to_AR_parameters(noise_power_spectrum_estimate,AR_model_order_noise);
    
    
    % y=darkalm3c(xw,as,bs,cs,an,bn,cn) ;
    [y_signal_estimate, total_forward_prediction_error] = ...
        get_clean_signal_estimate_using_two_pass_kalman_filter(current_frame,...
            AR_paraemters_speech,AR_gain_speech,auto_correlation_sequence_speech,...
            AR_parameters_noise,AR_gain_noise,auto_correlation_sequence_noise);
    
    %Overlap Add:
    output_signal(start_index:stop_index) = output_signal(start_index:stop_index) ...
        + hanning_window'.*y_signal_estimate ;
end ;


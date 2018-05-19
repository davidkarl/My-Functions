%test using signal diversity using turbulence files
% clear all;

%close previous instances:
try
   fclose(fid_turbulence);
   fclose(fid_speech);
   fclose('all');
   release(fractional_delay_object);
   release(audio_player_object);
   release(audio_file_writer_demodulated);
   release(analytic_signal_object);
   release(analytic_signal_object2);
   release('all');
catch
end 
 
%GET ANALYTIC SIGNAL TO INSERT THE SIGNAL INTO ITS PHASE:
file_name='turbulence_1400pm_new';
directory='C:\Users\master\Desktop\matlab\sol turbulence experiment\sound bin files for speech enhancement purposes';
file_name_with_bin = strcat(file_name,'.bin');
full_file_name_turbulence = fullfile(directory,file_name_with_bin);

%GET SPEECH SIGNAL TO INSERT INTO THE ANALYTIC SIGNAL PHASE:
file_name='10_second_speech_60dBspl';
directory='C:\Users\master\Desktop\matlab\sol turbulence experiment\sound bin files for speech enhancement purposes';
file_name_with_bin = strcat(file_name,'.bin');
full_file_name_speech = fullfile(directory,file_name_with_bin);

%initialize analytic signal fid:
fid_turbulence = fopen(full_file_name_turbulence,'r');
number_of_elements_in_file_turbulence = fread(fid_turbulence,1,'double');
Fs = fread(fid_turbulence,1,'double');
down_sample_factor = round(Fs/44100);
Fs_downsampled = Fs/down_sample_factor;

%initialize speech signal fid:
fid_speech = fopen(full_file_name_speech,'r');
number_of_elements_in_file_speech = fread(fid_speech,1,'double');
Fs_speech = fread(fid_speech,1,'double');
down_sample_factor = round(Fs_speech/44100);
Fs_downsampled_speech = Fs_speech/down_sample_factor;
 
%Multi-Channel properties:
number_of_channels = 4;

%Decide on number of samples per frame:
time_for_FFT_frame = 1; %100[mSec]
samples_per_frame = round(Fs*time_for_FFT_frame);
number_of_frames_in_file_speech = floor(number_of_elements_in_file_speech/samples_per_frame);
number_of_frames_in_file_turbulence = floor(number_of_elements_in_file_turbulence/samples_per_frame);

%decide on speech peak value:
speech_peak_value = 0.2;
counter=1;
peak_current_value = 0; 
while counter<number_of_frames_in_file_speech-4
    current_frame = fread(fid_speech,samples_per_frame,'double');
    peak_current_value = max(peak_current_value,max(abs(current_frame)));
    counter=counter+1; 
end
speech_scaling = peak_current_value/speech_peak_value;
frewind(fid_speech);
bla = fread(fid_speech,2,'double');


%Default Initial Values:
flag_PM_or_FM=1;
if flag_PM_or_FM==1
   PM_FM_str='PM'; 
else
   PM_FM_str='FM';
end
    
%audio player object:
audio_player_object = dsp.AudioPlayer;
audio_player_object.SampleRate = 44100;
audio_player_object.QueueDuration = 3;

%analytic signal object:
analytic_signal_object = dsp.AnalyticSignal;
analytic_signal_object.FilterOrder = 100;
analytic_signal_object2 = dsp.AnalyticSignal;
analytic_signal_object2.FilterOrder = 100;

%basic filter parameters:
Fc = 12000; %initial Fc
carrier_filter_parameter=20;
signal_filter_parameter=20;
carrier_filter_length = 128*10;
signal_filter_length = 128*10;
%signal filter:
filter_name_signal = 'hann';
signal_filter_type = 'bandpass';
signal_start_frequency = 150;
signal_stop_frequency = 3000;
[signal_filter] = get_filter_1D(filter_name_signal,signal_filter_parameter,signal_filter_length,Fs,signal_start_frequency,signal_stop_frequency,signal_filter_type);
signal_filter_object = dsp.FIRFilter('Numerator',signal_filter.Numerator);    
signal_filter_object2 = dsp.FIRFilter('Numerator',signal_filter.Numerator); 
%carrier filter:
BW = signal_stop_frequency*2;
filter_name_carrier = 'hann';
f_low_cutoff_carrier = Fc-BW/2;
f_high_cutoff_carrier = Fc+BW/2;
carrier_filter_type = 'bandpass';
[carrier_filter] = get_filter_1D(filter_name_carrier,carrier_filter_parameter,carrier_filter_length,Fs,f_low_cutoff_carrier,f_high_cutoff_carrier,carrier_filter_type);
carrier_filter_object = dsp.FIRFilter('Numerator',carrier_filter.Numerator);

%Smoothing and Despiking parameters:
smoother_derivative_weight = 0.6; 
smoother_differentiation_order = 2;
click_threshold = 0.00;
fraction_of_average_RMS_to_be_deemed_click = 0.05; 
universal_criterion_multiple = 0.7;


%save demodulation to wav file if wanted:
flag_save_to_wav=1;
if flag_save_to_wav==1
    audio_file_writer_demodulated = dsp.AudioFileWriter;
%     audio_file_writer_demodulated.Filename = strcat(fullfile(directory,file_name), PM_FM_str ,' final demodulated audio ', ' ', num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),', number of receivers = ',num2str(number_of_channels),', [Hz]','.wav');
    audio_file_writer_demodulated.Filename = strcat('strong turbulence ', num2str(number_of_channels),...
        ' channels simple average strong sound without despiking','.wav');
    audio_file_writer_demodulated.SampleRate = 44100;
end

%fractional delay input:
fractional_delay_object = dsp.VariableFractionalDelay;

%get fractional delays of the difference segments of turbulence files:
basic_delay_in_seconds_between_each_channel = time_for_FFT_frame;
basic_delay_in_samples_between_each_channel = floor(basic_delay_in_seconds_between_each_channel*Fs);
basic_delay_in_bytes_between_each_channel = basic_delay_in_samples_between_each_channel * 8;
for channel_counter = 1:number_of_channels
    delay_in_seconds_between_frames_current = basic_delay_in_seconds_between_each_channel * (channel_counter-1);
    delay_in_seconds_between_frames(channel_counter) = delay_in_seconds_between_frames_current;
    delay_in_samples_between_frames(channel_counter) = floor(delay_in_seconds_between_frames_current * Fs);
    fractional_delay_needed(channel_counter) = delay_in_seconds_between_frames_current*Fc - floor(delay_in_seconds_between_frames_current*Fc);
end

counter=1;
decimator_object = dsp.FIRDecimator(down_sample_factor);

%Initialize vecs:
current_frame_delayed_signals = zeros(number_of_channels, samples_per_frame);
current_frame_analytic_signals = zeros(number_of_channels, samples_per_frame);
analytic_signals_after_carrier_removal = zeros(number_of_channels, samples_per_frame);
phase_signals_after_carrier_removal = zeros(number_of_channels, samples_per_frame);
filtered_phase_signals = zeros(number_of_channels, samples_per_frame);
first_phase_current_vec = zeros(number_of_channels,1); 
last_phase_previous_vec = zeros(number_of_channels,1);
big_frame = 0;
current_t_vec = make_row(my_linspace(1/Fs,1/Fs,samples_per_frame));
while counter<number_of_frames_in_file_speech-1
    tic
    
    %Get speech signal current frame and scale it:
    current_frame_speech = fread(fid_speech,samples_per_frame,'double'); 
    current_frame_speech = current_frame_speech/speech_scaling;
    
    %Get turbulence distance factor assuming increasing signal with distance:
    turbulence_current_distance = 200;
    turbulence_final_distance = 1000;
    turbulence_phase_factor = sqrt(turbulence_final_distance/turbulence_current_distance);
    
    %Get turblence signals current frames:
    double_to_bytes_factor = 8;
    for channel_counter = 1:number_of_channels
        %go forward a certain amount of samples for next channel:
        fseek(fid_turbulence, delay_in_samples_between_frames(channel_counter)*double_to_bytes_factor, 0);
        
        %read current frame from current channel:
        current_frame = fread(fid_turbulence, samples_per_frame, 'double');
        current_frame_delayed_signals(channel_counter,:) = step(fractional_delay_object, current_frame, fractional_delay_needed(channel_counter));
        current_frame_analytic_signal = step(analytic_signal_object, squeeze(current_frame_delayed_signals(channel_counter,:))' );
        current_frame_analytic_signals(channel_counter,:) = abs(current_frame_analytic_signal).*exp(1i*angle(current_frame_analytic_signal)).*exp(1i*current_frame_speech/turbulence_phase_factor);
        
        %rewind current frame samples_per_frame samples back:
        fseek(fid_turbulence,-samples_per_frame*double_to_bytes_factor,0);
        fseek(fid_turbulence,-delay_in_samples_between_frames(channel_counter)*double_to_bytes_factor,0);
    end
    %uptick by samples_per_frame to move things along:
    fseek(fid_turbulence, samples_per_frame*double_to_bytes_factor, 0);
    
    %Start Demodulation:
    %(1). create relevant t_vec:
%     current_t_vec = (counter-1)*(samples_per_frame/Fs) + (0:samples_per_frame-1)/Fs;
    current_t_vec = current_t_vec + samples_per_frame/Fs;
    for channel_counter = 1:number_of_channels
        
 
       %(2). multiply by proper phase term to get read of most of carrier:
       analytic_signals_after_carrier_removal(channel_counter,:) = ...
           current_frame_analytic_signals(channel_counter,:).*exp(-1i*2*pi*Fc*current_t_vec);
       analytic_signal_conj = conj(analytic_signals_after_carrier_removal(channel_counter,:));
       analytic_signal_abs = sqrt(analytic_signals_after_carrier_removal(channel_counter,:).*analytic_signal_conj);
       analytic_signal_abs_mean = rms(analytic_signal_abs);
       
       %(3). Turn analytic signal to FM:
       phase_signals_after_carrier_removal(channel_counter,2:end) = angle( analytic_signals_after_carrier_removal(channel_counter,2:end).*conj(analytic_signals_after_carrier_removal(channel_counter,1:end-1)) );
       first_phase_current_vec(channel_counter) = analytic_signals_after_carrier_removal(channel_counter,1);
       phase_signals_after_carrier_removal(channel_counter,1) = angle(first_phase_current_vec(channel_counter).*conj(last_phase_previous_vec(channel_counter)));
       last_phase_previous_vec(channel_counter) = analytic_signals_after_carrier_removal(channel_counter,end);
       
       
       %Remove Clicks options:
       flag_use_only_found_only_additional_or_both = 3;
       number_of_indices_around_spikes_to_delete_as_well = 2;
       additional_indices_to_interpolate_over = find( analytic_signal_abs < click_threshold | ...
           analytic_signal_abs < analytic_signal_abs_mean*fraction_of_average_RMS_to_be_deemed_click );
       flag_use_my_spline_or_matlabs_or_binary_or_do_nothing = 4;
       %Detect clicks clicks:
       flag_use_despiking_or_nothing = 1;
       phase_signal_difference = make_column(phase_signals_after_carrier_removal(channel_counter,:));
       if flag_use_despiking_or_nothing == 1
           [phase_signal_after_mask,indices_containing_spikes_logical_mask,~,~] = despike_SOL_fast_with_logical_masks(phase_signal_difference, flag_use_only_found_only_additional_or_both,additional_indices_to_interpolate_over, number_of_indices_around_spikes_to_delete_as_well, flag_use_my_spline_or_matlabs_or_binary_or_do_nothing,universal_criterion_multiple);
           if ~isempty(indices_containing_spikes_logical_mask)==1
               analytic_signal_abs(indices_containing_spikes_logical_mask) = 0;
           end
       else
           phase_signal_after_mask = phase_signal_difference;
       end
       
       %Smooth signal from spikes
       spike_deleted_phase_signal  = phase_signal_difference(:) .* not(indices_containing_spikes_logical_mask);
       flag_do_smoothing_or_not = 0; 
       if flag_do_smoothing_or_not==1 
           [phase_signal_after_mask_and_smoothing] = smooth_lsq_minimum_iterations_with_preweights2(phase_signal_after_mask(:),analytic_signal_abs',smoother_derivative_weight,smoother_differentiation_order);
       end
       
       %(6). Filter signal:
%        filtered_phase = step(signal_filter_object,phase_signal_after_mask_and_smoothing);
       
       %(7). Turn to PM if wanted 
       flag_PM_or_FM = 0;
       if flag_PM_or_FM==1
           filtered_phase = cumsum(filtered_phase);
       end 
       
       %(8). Assign filtered_phase to filtered_phase_signals:
%        filtered_phase2 = step(signal_filter_object2,filtered_phase(:));
       filtered_phase_signals(channel_counter,:) = phase_signal_difference';
       
    end %end of channel_counter loop
    
    %Get signal weights:
    flag_use_weights_or_simple_average = 1;
    moment_power = 2;
    if flag_use_weights_or_simple_average == 1
        current_signal_weights = abs(current_frame_analytic_signals).^moment_power;
    else
        current_signal_weights = ones(number_of_channels,samples_per_frame);
    end
    current_total_variance = sum(current_signal_weights,1);
    
    %(9). Weight signals or use switching:
    flag_weights_or_switching = 1; 
    if flag_weights_or_switching == 1
        current_weighted_signal = (sum(current_signal_weights.*filtered_phase_signals,1)) ./ current_total_variance;
    else
        [max_from_each_channel_analytic_signal,index] = max(abs(current_frame_analytic_signals),[],1);
        indices = index + my_linspace(0,number_of_channels,samples_per_frame);
        current_weighted_signal = filtered_phase_signals(indices);
    end
    
    %(10). use PCA if wanted:
    flag_use_PCA_instead_of_weights = 1;
    if flag_use_PCA_instead_of_weights==1
        [Zpca,T,U,mu] = myPCA(filtered_phase_signals',1);
        current_weighted_signal = U / T * Zpca + repmat(mu,1,number_of_channels);
        current_weighted_signal = mean(current_weighted_signal,2);
    end
    
    %(11). Downsample:
    final_phase = step(decimator_object,current_weighted_signal(:));   
       
    flag_save_to_wav = 0;
    multiplication_factor=10;
    if flag_save_to_wav==1
        step(audio_file_writer_demodulated,[final_phase(:),final_phase(:)]*multiplication_factor);
    end
    
    flag_sound_demodulation=1;
    if flag_sound_demodulation==1
        big_frame = [big_frame(:);final_phase(:)];
%         step(audio_player_object,[final_phase(:)]*multiplication_factor);
%         step(audio_player_object,[filtered_phase(:)]*multiplication_factor);
%         step(audio_player_object,[current_frame_speech(:)]*multiplication_factor);
    end
    
    counter=counter+1;
    toc
end

big_frame2 = filter(signal_filter,big_frame);
big_frame2 = cumsum(big_frame2(:));
big_frame2 = filter(signal_filter,big_frame2);
sound((big_frame),44100);

try
    fclose(fid_turbulence);
    fclose(fid_speech);
    fclose('all');
    release(fractional_delay_object);
    release(audio_player_object);
    release(audio_file_writer_demodulated);
    release(analytic_signal_object);
    release(analytic_signal_object2);
    release('all');
catch 
end








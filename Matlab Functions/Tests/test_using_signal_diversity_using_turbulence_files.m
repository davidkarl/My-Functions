%test using signal diversity using turbulence files
clear all;

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
time_for_FFT_frame = 0.1; %100[mSec]
samples_per_frame = round(Fs*time_for_FFT_frame);
number_of_frames_in_file_speech = floor(number_of_elements_in_file_speech/samples_per_frame);
number_of_frames_in_file_turbulence = floor(number_of_elements_in_file_turbulence/samples_per_frame);

%decide on speech peak value:
speech_peak_value = 0.3;
counter=1;
peak_current_value = 0; 
while counter<number_of_frames_in_file_speech-10
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
audio_player_object.QueueDuration = 2;

%analytic signal object:
analytic_signal_object = dsp.AnalyticSignal;
analytic_signal_object.FilterOrder = 100;
analytic_signal_object2 = dsp.AnalyticSignal;
analytic_signal_object2.FilterOrder = 100;

%basic filter parameters:
Fc = 12000; %initial Fc
BW = 6000; %initial BW;
carrier_filter_parameter=20;
signal_filter_parameter=20;
carrier_filter_length = 128*10;
signal_filter_length = 128*10;
%carrier filter:
filter_name_carrier = 'hann';
f_low_cutoff_carrier = Fc-BW/2;
f_high_cutoff_carrier = Fc+BW/2;
carrier_filter_type = 'bandpass';
[carrier_filter] = get_filter_1D(filter_name_carrier,carrier_filter_parameter,carrier_filter_length,Fs,f_low_cutoff_carrier,f_high_cutoff_carrier,carrier_filter_type);
carrier_filter_object = dsp.FIRFilter('Numerator',carrier_filter.Numerator);
%signal filter:
filter_name_signal = 'hann';
signal_filter_type = 'bandpass';
signal_start_frequency=200;
signal_stop_frequency=2000;
[signal_filter] = get_filter_1D(filter_name_signal,signal_filter_parameter,signal_filter_length,Fs,signal_start_frequency,signal_stop_frequency,signal_filter_type);
signal_filter_object = dsp.FIRFilter('Numerator',signal_filter.Numerator);    


%save demodulation to wav file if wanted:
flag_save_to_wav=0;
if flag_save_to_wav==1
    audio_file_writer_demodulated = dsp.AudioFileWriter;
    audio_file_writer_demodulated.Filename = strcat(fullfile(directory,file_name), PM_FM_str ,' final demodulated audio ', ' ', num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),', number of receivers = ',num2str(number_of_channels),', [Hz]','.wav');
    audio_file_writer_demodulated.SampleRate = 44100;
end

%fractional delay input:
fractional_delay_object = dsp.VariableFractionalDelay;

%get fractional delays of the difference segments of turbulence files:
basic_delay_in_seconds_between_each_channel = time_for_FFT_frame*10;
basic_delay_in_samples_between_each_channel = floor(basic_delay_in_seconds_between_each_channel*Fs);
basic_delay_in_bytes_between_each_channel = basic_delay_in_samples_between_each_channel * 8;
for channel_counter = 1:number_of_channels
    delay_in_seconds_between_frames_current = basic_delay_in_seconds_between_each_channel * (channel_counter-1);
    delay_in_seconds_between_frames(channel_counter) = delay_in_seconds_between_frames_current;
    delay_in_samples_between_frames(channel_counter) = floor(delay_in_seconds_between_frames_current * Fs);
    fractional_delay_needed(channel_counter) = delay_in_seconds_between_frames_current*Fc - floor(delay_in_seconds_between_frames_current*Fc);
end

counter=1;
multiplication_factor=2;
decimator_object = dsp.FIRDecimator(down_sample_factor);

%Initialize vecs:
current_frame_delayed_signals = zeros(number_of_channels, samples_per_frame);
current_frame_analytic_signals = zeros(number_of_channels, samples_per_frame);
current_big_frame = 0;
while counter<number_of_frames_in_file_speech-1
    tic
     
    %Get speech signal current frame and scale it:
    current_frame_speech = fread(fid_speech,samples_per_frame,'double'); 
    current_frame_speech = current_frame_speech/speech_scaling;
    
    %Get turbulence distance factor assuming increasing signal with distance:
    turbulence_current_distance = 200;
    turbulence_final_distance = 200;
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
    
    %Get signal weights:
    flag_use_per_sample_or_per_weight_or_simple_average = 1;
    if flag_use_per_sample_or_per_weight_or_simple_average == 1
        current_signal_weights = abs(current_frame_analytic_signals).^2;
    elseif flag_use_per_sample_or_per_weight_or_simple_average == 2
        current_signal_weights = rms(current_frame_delayed_signals,2).^2;
        current_signal_weights = repmat(current_signal_weights,[1,samples_per_frame]);
    elseif flag_use_per_sample_or_per_weight_or_simple_average == 3
        current_signal_weights = ones(number_of_channels,samples_per_frame);
    end
    current_total_variance = sum(current_signal_weights,1);
    
    %Weight signals or use switching ON ANALYTIC SIGNALS:
    flag_weights_or_switching = 1;
    if flag_weights_or_switching == 1
        current_weighted_analytic_signal = (sum(current_signal_weights.*current_frame_analytic_signals,1)) ./ current_total_variance;
    else
        [max_from_each_channel_analytic_signal,index] = max(abs(current_frame_analytic_signals),[],1);
        indices = index + my_linspace(0,number_of_channels,samples_per_frame);
        current_weighted_analytic_signal = current_frame_analytic_signals(indices);
    end
    
    %Start Demodulation ON COMBINED ANALYTIC SIGNALS:
    %(1). create relevant t_vec:
    current_t_vec = (counter-1)*(samples_per_frame/Fs) + (0:samples_per_frame-1)/Fs;
    %(2). multiply by proper phase term to get read of most of carrier:
    analytic_signal_after_carrier_removal = current_weighted_analytic_signal.*exp(-1i*2*pi*Fc*current_t_vec);
    %(3). Turn analytic signal to FM:
    phase_signal_after_carrier_removal = angle(analytic_signal_after_carrier_removal(2:end).*conj(analytic_signal_after_carrier_removal(1:end-1)));
    %(4). Remove left over DC:
    phase_signal_after_carrier_removal = phase_signal_after_carrier_removal - mean(phase_signal_after_carrier_removal);
    
    %Add last term from previous frame to create a frame of equal size as original raw frame:
    if counter == 1
        last_term = 0;
    end
    phase_signal_after_carrier_removal = [last_term, phase_signal_after_carrier_removal];
    last_term = phase_signal_after_carrier_removal(end);

    %Set thresholds and Build masks:
    click_threshold = 0.008/sqrt(number_of_channels);
    spike_peak_threshold = 2;
    %GET INDIVIDUAL MASKS INDICES TO KEEP:
    indices_to_disregard_click = find( (abs(analytic_signal_after_carrier_removal) < click_threshold));
    indices_to_disregard_phase = find( (abs(phase_signal_after_carrier_removal) > spike_peak_threshold));
    indices_to_disregard_total = unique(sort([indices_to_disregard_click(:);indices_to_disregard_phase(:)]));
    uniform_sampling_indices = (1:length(current_frame))';
    phase_signal_mask = (abs(phase_signal_after_carrier_removal)<spike_peak_threshold);
    analytic_signal_mask = (abs(analytic_signal_after_carrier_removal)>click_threshold);
    
    %(5).Remove Clicks:
    flag_use_despiking_or_just_my_mask_or_nothing = 1;
    flag_use_only_found_only_additional_or_both = 3;
    additional_indices_to_interpolate_over = indices_to_disregard_total;
    number_of_indices_around_spikes_to_delete_as_well = 0;
    flag_use_my_spline_or_matlabs_or_binary_or_do_nothing = 3;
    %use despiking if wanted:
    if flag_use_despiking_or_just_my_mask_or_nothing == 1
        [phase_signal_after_mask, indices_containing_spikes_expanded, ~,~,~] = despike_SOL(phase_signal_after_carrier_removal, flag_use_only_found_only_additional_or_both, additional_indices_to_interpolate_over, number_of_indices_around_spikes_to_delete_as_well, flag_use_my_spline_or_matlabs_or_binary_or_do_nothing);
    elseif flag_use_despiking_or_just_my_mask_or_nothing == 2
        phase_signal_after_mask = phase_signal_after_carrier_removal;
        phase_signal_after_mask = phase_signal_after_mask.*analytic_signal_mask;
        phase_signal_after_mask = phase_signal_after_mask.*phase_signal_mask;
    elseif flag_use_despiking_or_just_my_mask_or_nothing == 3
        phase_signal_after_mask = phase_signal_after_carrier_removal;
    end
    
    %(6). Filter signal:
    filtered_phase = step(signal_filter_object,phase_signal_after_carrier_removal');
    
    %(7). Turn to PM if wanted
    flag_PM_or_FM = 2;
    if flag_PM_or_FM==1
        filtered_phase = cumsum(filtered_phase);
    end
                
    %(8). Down sample:
    filtered_phase = step(decimator_object,filtered_phase);
    
    flag_save_to_wav = 0;
    if flag_save_to_wav==1
        step(audio_file_writer_demodulated,[filtered_phase(:),filtered_phase(:)]*multiplication_factor);
    end
        
    flag_sound_demodulation = 1;
    if flag_sound_demodulation==1
        step(audio_player_object,[filtered_phase(:)]*multiplication_factor);
    end

    
    
    counter=counter+1;
    toc
end



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








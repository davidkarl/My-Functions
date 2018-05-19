% function [output_arguments] = test_SOL_turbulence_recording(directory,raw_file_name)
% test SOL turbulence recording:

%1=from workspace, 2=from binary file:
flag_signal_source=2;
flag_use_despiking = 0;

base_directory = 'C:\Users\master\Desktop\matlab\SIVSOL';
file_names = search_files(base_directory,'*.bin',0,1);
for k=1:length(file_names)
    file_names{k} = get_only_file_name_from_full_path(file_names{k},0);
end

%use test_SOL_turbulence_recording:
for tt=1:length(file_names)
    directory=base_directory;
    raw_file_name = file_names{tt}; 
    tic 
    
try
    fclose(fid);
    fclose(fid_amplitude);
    fclose(fid_phase);
    release(audio_player_object);
    release(audio_file_writer_demodulated);
    release(logger_object);
    release(audio_file_reader);
    release(audio_file_writer_raw);
    release(logger_object_compressed_and_clipped);
    release(compressed_and_clipped_analytic_signal_object);
    release(analytic_signal_object);
    release(phase_unwrapping_object);
    release(phase_unwrapping_object2);
    release(audio_file_writer_demodulated_compressed_and_clipped);
    fclose('all');
catch 
end

%Audio Source:
if flag_signal_source==1
    %workspace file:
    raw_file_name = 'dudy buffer4';
    if ~exist('current_buffer');
        load(raw_file_name);
    end
    signal_source_object = dsp.SignalSource(current_buffer(:), samples_per_frame);
    number_of_elements_in_file = length(current_buffer);
    Fs=44100*20;
else 
    %binary file:
    try 
       fclose(fid); 
    catch
        
    end
%     directory = 'C:\Users\master\Desktop\matlab\sol turbulence experiment\july 13th turbulence experiments\only good audio files';
%     raw_file_name = 'large_beams_170m_wall_50percent_DC04_1mm_mask_carrier_1430pm2_audio';
    flag_use_predetermined_Fc=1;
    Fc=12000;
    file_name_with_bin = strcat(raw_file_name,'.bin');
    full_file_name = fullfile(directory,file_name_with_bin);
    fid = fopen(full_file_name);
%     number_of_elements_in_file = fread(fid,1,'double');
%     Fs = fread(fid,1,'double');
    
    %FOR SIVSOL:
    Fs = 44100;
    number_of_elements_in_file = round(Fs*5);
end 
 
%Default Initial Values: 
time_for_FFT_frame = 0.1; %10[mSec]
counts_per_time_frame = round(Fs*time_for_FFT_frame);
samples_per_frame=counts_per_time_frame;
queue_duration=1;

%Get Transfer Function:
directory = 'C:\Users\master\Desktop\matlab\sol turbulence experiment\';
AM_raw_file_name = 'AM_calibration';
file_name_with_bin = strcat(AM_raw_file_name,'.bin');
full_file_name_AM = fullfile(directory,file_name_with_bin);
AM_fid = fopen(full_file_name_AM);
number_of_frequencies = fread(AM_fid,1,'double');
frequencies_vec = fread(AM_fid,number_of_frequencies,'double');
average_AM_values_vec = fread(AM_fid,number_of_frequencies,'double');
[f_vec_transfer_function] = fft_get_frequency_vec(samples_per_frame,Fs,0);
interpolated_transfer_function = zeros(length(f_vec_transfer_function),1);
logical_indices = (abs(f_vec_transfer_function)>=frequencies_vec(1) & abs(f_vec_transfer_function)<=frequencies_vec(end));
interpolated_transfer_function(logical_indices==1) = interp1(frequencies_vec,average_AM_values_vec,abs(f_vec_transfer_function(find(abs(f_vec_transfer_function)>=frequencies_vec(1) & abs(f_vec_transfer_function)<=frequencies_vec(end)))),'spline');
interpolated_transfer_function = normalize_signal(interpolated_transfer_function,1);
interpolated_transfer_function(logical_indices==0) = 1;
directory = 'C:\Users\master\Desktop\matlab\SIVSOL\';

%phase unwrapping object:
phase_unwrapping_object = dsp.PhaseUnwrapper;
phase_unwrapping_object2 = dsp.PhaseUnwrapper;

%analytic signal object:
analytic_signal_object = dsp.AnalyticSignal;
analytic_signal_object.FilterOrder=100;
compressed_and_clipped_analytic_signal_object = dsp.AnalyticSignal;
compressed_and_clipped_analytic_signal_object.FilterOrder = 100;
     
%maximum finder object:
maximum_finder_object = dsp.Maximum;

%fft object:
fft_object = dsp.FFT;
fft_object_phase = dsp.FFT;

%spectrum object:
spectrum_object = dsp.SpectrumEstimator;

%array adder object:
array_adder_object = dsp.ArrayVectorAdder;

%peak to peak object:
p2p_object = dsp.PeakToPeak;

%rms object:
rms_object = dsp.RMS;

% %audio player object:
down_sample_factor=2;
% audio_player_object = dsp.AudioPlayer;
% audio_player_object.SampleRate=Fs/down_sample_factor;
% audio_player_object.QueueDuration=2;

%log actual demodulated audio if wanted:
logger_object = dsp.SignalLogger;
logger_object_compressed_and_clipped = dsp.SignalLogger;

%Initialize Flags:
flag_only_calculate_simple_raw_statistics=0;
flag_sound_PM_or_FM=2;
flag_normalize_demodulation = 0;
flag_use_my_demodulation_or_dans = 2;
flag_try_compressed_and_clipped_carrier=0;
flag_show_FFT_stats_during_run=0;
flag_show_demodulated_PM_and_FM_phase_during_run=0;
flag_show_raw_vs_compressed_and_clipped_signal_during_run=0;
flag_log_demodualted_audio_to_workspace=0;
flag_sound_demodulation_later=0;
flag_sound_demodulation_during_loop_run=0;
flag_sound_raw_signal_or_compressed_and_clipped_signal = 1; %1 for raw, 2 for compressed and clipped
flag_write_binary_files_of_amplitude_and_phase_and_later = 0;
flag_record_downsampled_raw_file_to_wav_file = 0;
flag_record_demodulated_audio_to_wav_file = 0;
flag_analyze_during_run_and_after = 1;
flag_analyze_phase_rms_over_time = 1;
flag_save_plots = 1;
flag_should_i_do_any_demodulation = flag_log_demodualted_audio_to_workspace || flag_sound_demodulation_later || flag_sound_demodulation_during_loop_run...
    || flag_write_binary_files_of_amplitude_and_phase_and_later || flag_record_demodulated_audio_to_wav_file || flag_analyze_during_run_and_after || flag_analyze_phase_rms_over_time ...
    || flag_try_compressed_and_clipped_carrier || flag_only_calculate_simple_raw_statistics;


%Initialize parameters and vecs:
second_to_start_analyzing_from = 0;
second_to_stop_analyzing_at = 2000;
if second_to_stop_analyzing_at > floor(number_of_elements_in_file/Fs)
   second_to_stop_analyzing_at = floor(number_of_elements_in_file/Fs); 
end
number_of_frames = floor(((second_to_stop_analyzing_at-second_to_start_analyzing_from)*Fs)/samples_per_frame);
max_value=0;
CNR_energy_division_max = -50; %something arbitrary and low as a "zero"
average_spectrum_raw_signal = zeros(samples_per_frame,1);
average_spectrum_amplitude = zeros(samples_per_frame,1);
average_spectrum_phase_FM = zeros(samples_per_frame-1,1);
average_spectrum_phase_PM = zeros(samples_per_frame-1,1);
average_spectrum_raw_signal_weighted = zeros(samples_per_frame,1);
average_spectrum_amplitude_weighted = zeros(samples_per_frame,1);
average_spectrum_phase_FM_weighted = zeros(samples_per_frame-1,1);
average_spectrum_phase_PM_weighted = zeros(samples_per_frame-1,1);
PM_phase_current = zeros(samples_per_frame,1);
FM_phase_current = zeros(samples_per_frame,1);
max_spectrum = zeros(samples_per_frame,1);
weights_vec = zeros(number_of_frames,1);
peak_fourier_value_over_time_vec = zeros(number_of_frames,1);
peak_fourier_frequency_vec = zeros(number_of_frames,1);
raw_p2p_over_time_vec = zeros(number_of_frames,1);
raw_rms_over_time_vec = zeros(number_of_frames,1);
filtered_frame_rms_vec = zeros(number_of_frames,1);
raw_rms_vec = zeros(number_of_frames,1);
FM_phase_filtered_rms_over_time_vec = zeros(number_of_frames,1);
PM_phase_filtered_rms_over_time_vec = zeros(number_of_frames,1);
FM_phase_filtered_rms_over_time_after_mask_vec = zeros(number_of_frames,1);
PM_phase_filtered_rms_over_time_after_mask_vec = zeros(number_of_frames,1);
PM_phase_raw_rms_over_time_vec = zeros(number_of_frames,1);
FM_phase_raw_rms_over_time_vec = zeros(number_of_frames,1);
PM_phase_raw_rms_over_time_after_mask_vec = zeros(number_of_frames,1);
FM_phase_raw_rms_over_time_after_mask_vec = zeros(number_of_frames,1);
PSD_total_energy_over_time_vec = zeros(number_of_frames,1);
fft_noise_floor_mean_over_time_vec = zeros(number_of_frames,1);
carrier_lobe_energy_over_time_vec = zeros(number_of_frames,1);
CNR_energy_division_over_time_log_vec = zeros(number_of_frames,1);
CNR_above_noise_floor_over_time_log_vec = zeros(number_of_frames,1);
BW_bins_around_mean_carrier_frequency = [10,50,100,200,400,500,1000,2000,5000,100000];
fft_energy_in_frequency_bins = zeros(length(BW_bins_around_mean_carrier_frequency),number_of_frames);
current_fft_max_value=0;
number_of_spikes_vec = zeros(number_of_frames,1);
if flag_try_compressed_and_clipped_carrier==1
    max_value_compressed_and_clipped=0;
    average_spectrum_raw_signal_compressed_and_clipped = zeros(samples_per_frame,1);
    average_spectrum_amplitude_compressed_and_clipped = zeros(samples_per_frame,1);
    average_spectrum_phase_compressed_and_clipped = zeros(samples_per_frame,1);
    max_spectrum_compressed_and_clipped = zeros(samples_per_frame,1);
    weights_vec_compressed_and_clipped = zeros(number_of_frames,1);
    peak_fourier_value_compressed_and_clipped_vec = zeros(number_of_frames,1);
    peak_fourier_frequency_compressed_and_clipped_vec = zeros(number_of_frames,1);
    p2p_compressed_and_clipped_vec = zeros(number_of_frames,1);
    rms_compressed_and_clipped_vec = zeros(number_of_frames,1);
    FM_phase_rms_compressed_and_clipped_vec = zeros(number_of_frames,1);
    PM_phase_rms_compressed_and_clipped_vec = zeros(number_of_frames,1);
    total_PSD_energy_compressed_and_clipped = zeros(number_of_frames,1);
    fft_energy_in_frequency_bins_compressed_and_clipped = zeros(length(BW_bins_around_mean_carrier_frequency),number_of_frames);
    current_fft_max_value_compressed_and_clipped=0;
    CNR_energy_division_over_time_compressed_vec = zeros(number_of_frames,1);
    CNR_above_noise_floor_over_time_compressed_vec = zeros(number_of_frames,1);
                
end

%binary writer object (for unwrapped phase and amplitude of analytic signal, and raw signal):
%i do this to be able to see later on graphs the entire signals while space on disk
%during run and later because wav file is compressed.
amplitude_file_name_to_be_written = strcat(raw_file_name,' analytic signal amplitude.bin');
phase_file_name_to_be_written = strcat(raw_file_name,' analytic signal phase.bin');
new_directory = 'C:\Users\master\Desktop\matlab\';
full_file_name_amplitude = fullfile(new_directory,amplitude_file_name_to_be_written);
full_file_name_phase = fullfile(new_directory,phase_file_name_to_be_written);
if flag_write_binary_files_of_amplitude_and_phase_and_later==1
    fid_amplitude = fopen(full_file_name_amplitude,'W+');
    fid_phase = fopen(full_file_name_phase,'W+');
    %put number of elements and Fs as first elements of the to-be binary file:
    fwrite(fid_amplitude,number_of_elements_in_file,'double');
    fwrite(fid_amplitude,Fs,'double');
    fwrite(fid_phase,number_of_elements_in_file,'double');
    fwrite(fid_phase,Fs,'double');
end

%record actual demodualted audio in wav file if wanted (the computer can more easily sound it live than binary):
if flag_record_demodulated_audio_to_wav_file==1
    audio_file_writer_demodulated = dsp.AudioFileWriter;
    audio_file_writer_demodulated.Filename = strcat(raw_file_name,' final demodulated audio.wav');
    audio_file_writer_demodulated.SampleRate = Fs/down_sample_factor;
    
    if flag_try_compressed_and_clipped_carrier==1
        audio_file_writer_demodulated_compressed_and_clipped = dsp.AudioFileWriter;
        audio_file_writer_demodulated_compressed_and_clipped.Filename = strcat(raw_file_name,' compressed and clipped final demodulated audio.wav');
        audio_file_writer_demodulated_compressed_and_clipped.SampleRate = Fs/down_sample_factor;
    end
end
%record raw file downsampled to wav file if wanted:
if flag_record_downsampled_raw_file_to_wav_file==1
    audio_file_writer_raw = dsp.AudioFileWriter;
    audio_file_writer_raw.Filename = strcat(raw_file_name,' raw file.wav');
    audio_file_writer_raw.SampleRate = Fs/down_sample_factor;
end

%Quickly go over file and search for average peak frequency (Fc):
counter=1; 
while counter<floor(number_of_elements_in_file/samples_per_frame)-1
    try
    tic
    %Get Frame:
    if flag_signal_source==1
        current_frame = step(signal_source_object);
        current_frame = current_frame(:,1);
    else
        current_frame = fread(fid,samples_per_frame,'double');
        if flag_record_downsampled_raw_file_to_wav_file==1
           step(audio_file_writer_raw,downsample(current_frame,down_sample_factor)); 
        end
    end  
    %FFT: 
    fft_vec_raw = fftshift(step(fft_object,current_frame));
    fft_vec_raw(1:round(length(fft_vec_raw)/2),1) = 0;
    [current_fft_max_value,current_fft_max_index] = step(maximum_finder_object,abs(fft_vec_raw));
    peak_fourier_frequency_vec(counter) = fft_index_to_frequency(length(fft_vec_raw),Fs,current_fft_max_index);
    counter=counter+1;
    toc
    catch
       break; 
    end
end
try
    fclose(fid);
catch 
end
average_max_frequency = mean(peak_fourier_frequency_vec);
if flag_use_predetermined_Fc==1
   average_max_frequency=Fc; 
end
frequency_bins = [average_max_frequency - BW_bins_around_mean_carrier_frequency(:), average_max_frequency + BW_bins_around_mean_carrier_frequency(:)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%BUILD FILTERS:
BW=5000; %initial BW; 
effective_lobe_BW_one_sided = 200;
if average_max_frequency<BW && flag_use_predetermined_Fc==0
   average_max_frequency=Fs/3; %just put something which has no chance of causing problems with filters creation
end
carrier_filter_length = 128*10;
carrier_filter_parameter=20;

%carrier filter:
filter_name_carrier = 'hann';
f_low_cutoff_carrier = average_max_frequency-BW/2;
f_high_cutoff_carrier = average_max_frequency+BW/2;
carrier_filter_type = 'bandpass';
[carrier_filter] = get_filter_1D(filter_name_carrier,carrier_filter_parameter,carrier_filter_length,Fs,f_low_cutoff_carrier,f_high_cutoff_carrier,carrier_filter_type);
carrier_filter_object = dsp.FIRFilter('Numerator',carrier_filter.Numerator);
%another carrier filter for compressed and clipped signal:
compressed_and_clipped_carrier_filter_object = dsp.FIRFilter('Numerator',carrier_filter.Numerator);
%signal filter:
signal_filter_length = 128*10;
signal_filter_parameter = 20;
signal_start_frequency = 300;
signal_stop_frequency = BW/2;
filter_name_signal = 'hann';
signal_filter_type = 'bandpass';
[signal_filter] = get_filter_1D(filter_name_signal,signal_filter_parameter,signal_filter_length,Fs,signal_start_frequency,signal_stop_frequency,signal_filter_type);
signal_filter_object1 = dsp.FIRFilter('Numerator',signal_filter.Numerator);    
signal_filter_object2 = dsp.FIRFilter('Numerator',signal_filter.Numerator);   
signal_filter_object3 = dsp.FIRFilter('Numerator',signal_filter.Numerator);   
signal_filter_object4 = dsp.FIRFilter('Numerator',signal_filter.Numerator);   
signal_filter_object5 = dsp.FIRFilter('Numerator',signal_filter.Numerator);   

%Compression and Clipping attributes:
pre_gain_compression_factor = 10000;
clipping_limit = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%GO over signal and do a lot of parameter recordings but first initialize signal source needed:
counter=1;
if flag_signal_source==1
    signal_source_object = dsp.SignalSource(current_buffer(:), samples_per_frame);
else
    fid = fopen(full_file_name);
    bla = fread(fid,2,'double'); %skip first two irrelevant elements
    bla = fread(fid,round(Fs*second_to_start_analyzing_from),'double'); %skip a few seconds according to variable
end


%Start actual main loop:
if flag_should_i_do_any_demodulation==1
    while counter<number_of_frames-1
%         try
%         tic
        %Get Frame:
        if flag_signal_source==1
            current_frame = step(signal_source_object);
            current_frame = current_frame(:,1);
        else 
            current_frame = fread(fid,samples_per_frame,'double');
        end 
        
        
        %Find Amplitude Vec and Phase Vec Over Time:
        %Filter carrier:
        filtered_carrier = step(carrier_filter_object,current_frame);   
        %Extract analytic signal from raw signal:
        analytic_signal = step(analytic_signal_object,filtered_carrier);
        
        
        if flag_only_calculate_simple_raw_statistics==1
            %ONLY calculate simple statistics for some purpose and thus go
            %over the file very quickly:
            raw_rms_vec(counter) = step(rms_object,current_frame);
            filtered_frame_rms_vec(counter) = step(rms_object,filtered_carrier);
            fft_noise_floor_mean_over_time_vec(counter) = fft_calculate_mean_between_frequencies(fft_vec_raw,Fs,2,1,1,[average_max_frequency+effective_lobe_BW_one_sided,average_max_frequency+BW]);
            
        else
            %ANALYZE THE WHOLE THING:
            
            %Extract analytic signal from compressed and clipped signal:
            if flag_try_compressed_and_clipped_carrier==1
                current_frame_compressed_and_clipped = current_frame*pre_gain_compression_factor;
                current_frame_compressed_and_clipped(current_frame_compressed_and_clipped>1) = 1;
                current_frame_compressed_and_clipped(current_frame_compressed_and_clipped<-1) = -1;
                filtered_carrier_compressed_and_clipped = step(compressed_and_clipped_carrier_filter_object, current_frame_compressed_and_clipped);
                analytic_signal_compressed_and_clipped = step(compressed_and_clipped_analytic_signal_object,filtered_carrier_compressed_and_clipped);
            end

            %Extract phase and amplitude signal from analytic signal in order to analyze them if wanted:
            if flag_write_binary_files_of_amplitude_and_phase_and_later==1 || flag_analyze_during_run_and_after==1
                %Extract (uncorrected) angle of analytic signal:
                phase_signal = step(phase_unwrapping_object,angle(analytic_signal));
                %Extract amplitude of analytic signal:
                amplitude_signal = abs(analytic_signal);

                if flag_try_compressed_and_clipped_carrier==1
                   phase_signal_compressed_and_clipped = step(phase_unwrapping_object,angle(analytic_signal_compressed_and_clipped));
                   amplitude_signal_compressed_and_clipped = abs(analytic_signal_compressed_and_clipped);
                end
            end

            %record analytic signal and phase signal to binary file if wanted:
            if flag_write_binary_files_of_amplitude_and_phase_and_later==1
                fwrite(fid_amplitude,amplitude_signal,'double');
                fwrite(fid_phase,phase_signal,'double');
            end

            %DO things with the PHASE extracted from analytic signal:
            if flag_record_demodulated_audio_to_wav_file==1 || flag_sound_demodulation_during_loop_run==1 || flag_show_FFT_stats_during_run==1 || flag_show_demodulated_PM_and_FM_phase_during_run==1 || flag_log_demodualted_audio_to_workspace==1 || flag_analyze_phase_rms_over_time==1

                %DEMODULATE RAW SIGNAL:
                if flag_use_my_demodulation_or_dans==1
                   %USE MY DEMODULATION METHOD:
                    %(1). creating relevant t_vec:
                    current_t_vec = (counter-1)*(samples_per_frame/Fs) + (0:samples_per_frame-1)/Fs;
                    %(2). multiply by proper phase term to get read of most of carrier:
                    analytic_signal_corrected = analytic_signal.*exp(-1i*2*pi*average_max_frequency*current_t_vec');
                    %(3). unwrap resulting signal:
                    phase_signal_corrected = step(phase_unwrapping_object2,angle(analytic_signal_corrected));
                    %(4). down sample:
                    down_sampled_phase = downsample(phase_signal_corrected,down_sample_factor);
                    PM_phase_current = phase_signal_corrected; 
                    FM_phase_current = phase_signal_corrected(2:end)-phase_signal_corrected(1:end-1);
                    %(5). filter unwrapped signal:
                    filtered_phase = step(signal_filter_object1,down_sampled_phase); 
                    %(6). Turn to FM if wanted
                    if flag_sound_PM_or_FM==2; filtered_phase = filtered_phase(2:end)-filtered_phase(1:end-1); end 
                    %(7). Normalize demodulation if wanted:
                    if flag_normalize_demodulation==1; filtered_phase = filtered_phase/max(abs(filtered_phase)); end
                elseif flag_use_my_demodulation_or_dans==2
                    %USE DAN'S DEMODULATINO METHOD:
                    %(1). creating relevant t_vec:
                    current_t_vec = (counter-1)*(samples_per_frame/Fs) + (0:samples_per_frame-1)/Fs;
                    %(2). multiply by proper phase term to get read of most of carrier:
                    analytic_signal_corrected = analytic_signal.*exp(-1i*2*pi*average_max_frequency*current_t_vec');
                    %(3). Turn analytic signal to FM:
                    phase_signal_corrected = angle(analytic_signal_corrected(2:end).*conj(analytic_signal_corrected(1:end-1)));
                    
                    %(4). Despike if wanted:
                    %Set thresholds and Build masks:
                    click_threshold = 0.000;
                    spike_peak_threshold = 2;
                    %Get individual mask indices to get rid of:
                    indices_to_disregard_click = find( abs(analytic_signal_corrected) < click_threshold );
                    indices_to_disregard_phase = find( abs(phase_signal_corrected) > spike_peak_threshold );
                    indices_to_disregard_total = unique(sort([indices_to_disregard_click(:);indices_to_disregard_phase(:)]));
                    uniform_sampling_indices = (1:length(current_frame))';
                    indices_to_keep_total = ismember(uniform_sampling_indices, indices_to_disregard_total);
                    
                    %FIND ANALYTIC SIGNAL AND PHASE SIGNAL MASKS:
                    phase_signal_mask = (abs(phase_signal_corrected)<spike_peak_threshold);
                    analytic_signal_mask = (abs(analytic_signal_corrected)>click_threshold);
                      
                    %Remove Clicks options:
                    flag_use_only_found_only_additional_or_both = 4;
                    additional_indices_to_interpolate_over = indices_to_disregard_total;
                    number_of_indices_around_spikes_to_delete_as_well = 0;
                    flag_use_my_spline_or_matlabs_or_binary_or_do_nothing = 3;
%                     [phase_signal_after_mask, indices_containing_spikes_expanded, ~,~,~] = despike_SOL(phase_signal_corrected', flag_use_only_found_only_additional_or_both, additional_indices_to_interpolate_over, number_of_indices_around_spikes_to_delete_as_well, flag_use_my_spline_or_matlabs_or_binary_or_do_nothing);
                    [phase_signal_after_mask,indices_containing_spikes_expanded,~,~] = despike_SOL_fast_with_logical_masks(phase_signal_corrected(:), flag_use_only_found_only_additional_or_both,additional_indices_to_interpolate_over, number_of_indices_around_spikes_to_delete_as_well, flag_use_my_spline_or_matlabs_or_binary_or_do_nothing,1);
                    phase_signal_after_mask = phase_signal_after_mask';
                    
                    %(5). down sample:
                    down_sampled_phase = downsample(phase_signal_corrected,down_sample_factor);
                    FM_phase_current = phase_signal_corrected; 
                    PM_phase_current = cumsum(FM_phase_current);
                    down_sampled_phase_after_mask = downsample(phase_signal_after_mask,down_sample_factor);
                    FM_phase_after_mask_current = phase_signal_after_mask;
                    PM_phase_after_mask_current = cumsum(FM_phase_after_mask_current);
                     
                    %(6). filter unwrapped signal:
                    filtered_phase = step(signal_filter_object1,FM_phase_current); 
                    
                    %filter phases for later registering:
                    FM_phase_after_mask_filtered_current = step(signal_filter_object2,FM_phase_after_mask_current);
                    PM_phase_after_mask_filtered_current = cumsum(FM_phase_after_mask_filtered_current);
                    FM_phase_filtered_current = step(signal_filter_object4,FM_phase_current);
                    PM_phase_filtered_current = cumsum(FM_phase_filtered_current);
                    
%                     figure
%                     plot(FM_phase_current);
%                     hold on;
%                     plot(FM_phase_after_mask_current,'g');
%                     close(gcf);
                    
                    %(7). Turn to PM if wanted
                    if flag_sound_PM_or_FM==1; filtered_phase = cumsum(filtered_phase); end    
                    %(8). Normalize demodulation if wanted:
                    if flag_normalize_demodulation==1; filtered_phase = filtered_phase/max(abs(filtered_phase)); end
                end      

                %DEMODULATE COMPRESSED AND CLIPPED SIGNAL:
                if flag_try_compressed_and_clipped_carrier==1
                    if flag_use_my_demodulation_or_dans==1
                       %USE MY DEMODULATION METHOD:
                        %(2). multiply by proper phase term to get read of most of carrier:
                        analytic_signal_corrected_compressed_and_clipped = analytic_signal_compressed_and_clipped.*exp(-1i*2*pi*average_max_frequency*current_t_vec');
                        %(3). unwrap resulting signal:
                        phase_signal_corrected_compressed_and_clipped = step(phase_unwrapping_object2,angle(analytic_signal_corrected_compressed_and_clipped));
                        %(4). down sample:
                        down_sampled_phase_compressed_and_clipped = downsample(phase_signal_corrected_compressed_and_clipped,down_sample_factor);
                        %(5). filter unwrapped signal:
                        filtered_phase_compressed_and_clipped = step(signal_filter_object1,down_sampled_phase_compressed_and_clipped); 
                        %(6). Turn to FM if wanted
                        if flag_sound_PM_or_FM==2; filtered_phase_compressed_and_clipped = filtered_phase_compressed_and_clipped(2:end)-filtered_phase_compressed_and_clipped(1:end-1); end 
                        %(7). Normalize demodulation if wanted:
                        if flag_normalize_demodulation==1; filtered_phase_compressed_and_clipped = filtered_phase_compressed_and_clipped/max(abs(filtered_phase_compressed_and_clipped)); end
                    elseif flag_use_my_demodulation_or_dans==2
                        %USE DAN'S DEMODULATINO METHOD:
                        %(2). multiply by proper phase term to get read of most of carrier:
                        analytic_signal_corrected_compressed_and_clipped = analytic_signal_compressed_and_clipped.*exp(-1i*2*pi*average_max_frequency*current_t_vec');
                        %(3). Turn analytic signal to FM:
                        phase_signal_corrected_compressed_and_clipped = angle(analytic_signal_corrected_compressed_and_clipped(2:end).*conj(analytic_signal_corrected_compressed_and_clipped(1:end-1)));
                        %(4). down sample:
                        down_sampled_phase_compressed_and_clipped = downsample(phase_signal_corrected_compressed_and_clipped,down_sample_factor);
                        %(5). filter unwrapped signal:
                        filtered_phase_compressed_and_clipped = step(signal_filter_object1,down_sampled_phase_compressed_and_clipped); 
                        %(6). Turn to PM if wanted
                        if flag_sound_PM_or_FM==1; filtered_phase_compressed_and_clipped = cumsum(filtered_phase_compressed_and_clipped); end    
                        %(7). Normalize demodulation if wanted:
                        if flag_normalize_demodulation==1; filtered_phase_compressed_and_clipped = filtered_phase_compressed_and_clipped/max(abs(filtered_phase_compressed_and_clipped)); end
                    end   
                end 

                if flag_record_demodulated_audio_to_wav_file==1
                    %record filtered signal:
                    step(audio_file_writer_demodulated,filtered_phase);
                    if flag_try_compressed_and_clipped_carrier==1
                       step(audio_file_writer_demodulated_compressed_and_clipped,filtered_phase_compressed_and_clipped); 
                    end
                end
                if flag_sound_demodulation_during_loop_run==1
                    %sound filtered signal:
                    if flag_sound_raw_signal_or_compressed_and_clipped_signal==1
                        step(audio_player_object,filtered_phase);
                    elseif flag_sound_raw_signal_or_compressed_and_clipped_signal==2
                        step(audio_player_object,filtered_phase_compressed_and_clipped);
                    end
                end
                if flag_log_demodualted_audio_to_workspace==1
                   %log filtered signal:
                   step(logger_object,filtered_phase);
                   if flag_try_compressed_and_clipped_carrier==1
                      step(logger_object_compressed_and_clipped,filtered_phase_compressed_and_clipped); 
                   end
                end
                if flag_analyze_phase_rms_over_time==1 && counter>1
                   %get phase RMS for both PM and FM:
                   PM_phase_raw_rms_over_time_vec(counter-1) = rms(PM_phase_current);
                   FM_phase_raw_rms_over_time_vec(counter-1) = rms(FM_phase_current);
                   PM_phase_raw_rms_over_time_after_mask_vec(counter-1) = rms(PM_phase_after_mask_current);
                   FM_phase_raw_rms_over_time_after_mask_vec(counter-1) = rms(FM_phase_after_mask_current);
                   PM_phase_filtered_rms_over_time_vec(counter-1) = rms(PM_phase_filtered_current);
                   FM_phase_filtered_rms_over_time_vec(counter-1) = rms(FM_phase_filtered_current);
                   PM_phase_filtered_rms_over_time_after_mask_vec(counter-1) = rms(PM_phase_after_mask_filtered_current);
                   FM_phase_filtered_rms_over_time_after_mask_vec(counter-1) = rms(FM_phase_after_mask_filtered_current);

                   if flag_try_compressed_and_clipped_carrier==1
                       if flag_sound_PM_or_FM==1
                           PM_phase_rms_compressed_and_clipped_vec(counter-1) = rms(filtered_phase_compressed_and_clipped);
                           FM_phase_rms_compressed_and_clipped_vec(counter-1) = rms(filtered_phase_compressed_and_clipped(2:end)-filtered_phase_compressed_and_clipped(1:end));
                       else 
                           PM_phase_rms_compressed_and_clipped_vec(counter-1) = rms(cumsum(filtered_phase_compressed_and_clipped));
                           FM_phase_rms_compressed_and_clipped_vec(counter-1) = rms(filtered_phase_compressed_and_clipped);
                       end 
                   end
                end
            end 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



            %Analyze carrier, amplitude and phase spectrum over time:
            if flag_analyze_during_run_and_after==1

                %Calculate and Assign FFTs for later analysis:
                fft_vec_phase_FM_raw = fftshift(step(fft_object_phase,FM_phase_current));
                fft_vec_phase_PM_raw = fftshift(step(fft_object_phase,PM_phase_current));
                fft_vec_amplitude = fftshift(step(fft_object,amplitude_signal));
                fft_vec_raw = fftshift(step(fft_object,current_frame));

                %Find Max fourier frequency and value and calculate fft noise floor:
                if flag_use_predetermined_Fc==0
                    [current_fft_max_value,current_fft_max_index] = step(maximum_finder_object,abs(fft_vec_raw));
                elseif flag_use_predetermined_Fc==1
                    current_fft_max_index = fft_frequency_to_index(length(fft_vec_raw),Fs,Fc,2);
                    current_fft_max_value = abs(fft_vec_raw(current_fft_max_index));
                end
                peak_fourier_value_over_time_vec(counter) = current_fft_max_value;
                peak_fourier_frequency_vec(counter) = fft_index_to_frequency(length(fft_vec_raw),Fs,current_fft_max_index);
                fft_noise_floor_mean_over_time_vec(counter) = fft_calculate_mean_between_frequencies(fft_vec_raw,Fs,2,1,1,[average_max_frequency+effective_lobe_BW_one_sided,average_max_frequency+BW]);
                
                %Calculate current CNR to see its characteristics:
                BW = 5000;
                current_spectrum = abs(fft_vec_raw).^2;
                [CNR_energy_division_current,CNR_above_noise_floor_with_RBW1Hz_current,CNR_above_noise_floor_with_RBW1Hz_vec_current,mean_noise_floor_PSD_current,carrier_max_current,carrier_lobe_energy_current] = ...
                     calculate_CNR(current_spectrum,3,BW,effective_lobe_BW_one_sided,Fs,average_max_frequency);
                carrier_lobe_energy_over_time_vec(counter) = carrier_lobe_energy_current;
                CNR_energy_division_over_time_log_vec(counter) = CNR_energy_division_current;
                CNR_above_noise_floor_over_time_log_vec(counter) = CNR_above_noise_floor_with_RBW1Hz_current;

                %Register CNR is current FFT peak is at the max it has been to see how it matches average and AM CNR:
                if CNR_energy_division_max < CNR_energy_division_current
                    %update max fft value:
                    max_value = current_fft_max_value;
                    %current spectrum:
                    max_spectrum = current_spectrum;
                    %get max CNR:
                    CNR_energy_division_max = CNR_energy_division_current;
                    CNR_above_noise_floor_with_RBW1Hz_max = CNR_above_noise_floor_with_RBW1Hz_current;
                    CNR_above_noise_floor_with_RBW1Hz_vec_max = CNR_above_noise_floor_with_RBW1Hz_vec_current;
                    mean_noise_floor_PSD_max = mean_noise_floor_PSD_current;
                    carrier_max_max = carrier_max_current;               
                end

                %Find Average Spectrum Per Frame:
                average_spectrum_raw_signal = (step(array_adder_object,average_spectrum_raw_signal , abs(fft_vec_raw).^2));
                average_spectrum_amplitude = (step(array_adder_object,average_spectrum_amplitude , abs(fft_vec_amplitude).^2));
                average_spectrum_phase_FM = (step(array_adder_object,average_spectrum_phase_FM , abs(fft_vec_phase_FM_raw).^2));
                average_spectrum_phase_PM = (step(array_adder_object,average_spectrum_phase_PM , abs(fft_vec_phase_PM_raw).^2));
                
                %Assign Weights to fft's according to fourier peak to show later an average
                %spectrum better reflecting the spectrum when we actually have a signal:
                weights_vec(counter) = current_fft_max_value^2;
                average_spectrum_raw_signal_weighted = (step(array_adder_object,average_spectrum_raw_signal_weighted , weights_vec(counter)*abs(fft_vec_raw).^2));
                average_spectrum_amplitude_weighted = (step(array_adder_object,average_spectrum_amplitude_weighted , weights_vec(counter)*abs(fft_vec_amplitude).^2));
                average_spectrum_phase_FM_weighted = (step(array_adder_object,average_spectrum_phase_FM_weighted , weights_vec(counter)*abs(fft_vec_phase_FM_raw).^2));
                average_spectrum_phase_PM_weighted = (step(array_adder_object,average_spectrum_phase_PM_weighted , weights_vec(counter)*abs(fft_vec_phase_PM_raw).^2));
                
                %Find Spectrum Energies over different BW's away from Carrier and Signal Energy over time:
                [current_fft_energy_in_frequency_bins] = fft_calculate_energy_between_frequencies(fft_vec_raw,Fs,2,2,1,frequency_bins);
                fft_energy_in_frequency_bins(:,counter) = current_fft_energy_in_frequency_bins;
                PSD_total_energy_over_time_vec(counter) = sum(abs(fft_vec_raw).^2);
                 
                %Find Raw (no filtering) Peak to Peak and RMS over time:
                raw_p2p_over_time_vec(counter) = step(p2p_object,current_frame);
                raw_rms_over_time_vec(counter) = step(rms_object,current_frame);
                filtered_frame_rms_vec(counter) = step(rms_object,filtered_carrier);
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            

                if flag_try_compressed_and_clipped_carrier==1
                    %Calculate and Assign FFTs (including correction by transfer function) for later analysis:
                    fft_vec_phase_compressed_and_clipped = fftshift(step(fft_object,phase_signal_compressed_and_clipped))./interpolated_transfer_function;
                    fft_vec_amplitude_compressed_and_clipped = fftshift(step(fft_object,amplitude_signal_compressed_and_clipped))./interpolated_transfer_function;
                    fft_vec_raw_compressed_and_clipped = fftshift(step(fft_object,current_frame_compressed_and_clipped))./interpolated_transfer_function;

                    %Find Max fourier frequency and value:
                    if flag_use_predetermined_Fc==0
                        [current_fft_max_value_compressed_and_clipped,current_fft_max_index_compressed_and_clipped] = step(maximum_finder_object,abs(fft_vec_raw_compressed_and_clipped));
                    elseif flag_use_predetermined_Fc==1
                        current_fft_max_index_compressed_and_clipped = fft_frequency_to_index(length(fft_vec_raw_compressed_and_clipped),Fs,Fc,2);
                        current_fft_max_value_compressed_and_clipped = abs(fft_vec_raw_compressed_and_clipped(current_fft_max_index_compressed_and_clipped));
                    end
                    peak_fourier_value_compressed_and_clipped_vec(counter) = current_fft_max_value_compressed_and_clipped;
                    peak_fourier_frequency_compressed_and_clipped_vec(counter) = fft_index_to_frequency(length(fft_vec_raw_compressed_and_clipped),Fs,current_fft_max_index_compressed_and_clipped);

                    %Calculate current CNR to see its characteristics:
                    BW = 5000;
                    current_spectrum_compressed_and_clipped = abs(fft_vec_raw_compressed_and_clipped).^2;
                    [CNR_current_compressed,CNR_above_noise_floor_with_RBW1Hz_current_compressed,CNR_above_noise_floor_with_RBW1Hz_vec_current_compressed,mean_noise_floor_current_compressed,carrier_max_current_compressed] = ...
                         calculate_CNR(current_spectrum_compressed_and_clipped,3,BW,effective_lobe_BW_one_sided,Fs,average_max_frequency);
                    CNR_energy_division_over_time_compressed_vec(counter) = CNR_energy_division_current;
                    CNR_above_noise_floor_over_time_compressed_vec(counter) = CNR_above_noise_floor_with_RBW1Hz_vec_current;

                    %Register CNR is current FFT peak is at the max it has been to see how it matches average and AM CNR:
                    if max_value_compressed_and_clipped < current_fft_max_value_compressed_and_clipped
                        %update max fft value:
                        max_value_compressed_and_clipped = current_fft_max_value_compressed_and_clipped;
                        %current spectrum:
                        max_spectrum_compressed_and_clipped = current_spectrum_compressed_and_clipped;

                        %get max CNR:
                        CNR_max_compressed_and_clipped = CNR_current_compressed;
                        CNR_above_noise_floor_with_RBW1Hz_max_compressed_and_clipped = CNR_above_noise_floor_with_RBW1Hz_current_compressed;
                        CNR_above_noise_floor_with_RBW1Hz_vec_max_compressed_and_clip = CNR_above_noise_floor_with_RBW1Hz_vec_current_compressed;
                        mean_noise_floor_max_compressed_and_clipped = mean_noise_floor_current_compressed;
                        carrier_max_max_compressed_and_clipped = carrier_max_current_compressed;
                    end

                    %Find Average Spectrum Per Frame:
                    average_spectrum_raw_signal_compressed_and_clipped = (step(array_adder_object,average_spectrum_raw_signal_compressed_and_clipped , abs(fft_vec_raw_compressed_and_clipped).^2));
                    average_spectrum_amplitude_compressed_and_clipped = (step(array_adder_object,average_spectrum_amplitude_compressed_and_clipped , abs(fft_vec_amplitude_compressed_and_clipped).^2));
                    average_spectrum_phase_compressed_and_clipped = (step(array_adder_object,average_spectrum_phase_compressed_and_clipped , abs(fft_vec_phase_compressed_and_clipped).^2));

                    %Assign Weights to fft's according to fourier peak to show later an average
                    %spectrum better reflecting the spectrum when we actually have a signal:
                    weights_vec_compressed_and_clipped(counter) = current_fft_max_value_compressed_and_clipped^2;
                    average_spectrum_raw_signal_weighted_compressed_and_clipped = (step(array_adder_object,average_spectrum_raw_signal_compressed_and_clipped , weights_vec_compressed_and_clipped(counter)*abs(fft_vec_raw_compressed_and_clipped).^2));
                    average_spectrum_amplitude_weighted_compressed_and_clipped = (step(array_adder_object,average_spectrum_amplitude_compressed_and_clipped , weights_vec_compressed_and_clipped(counter)*abs(fft_vec_amplitude_compressed_and_clipped).^2));
                    average_spectrum_phase_weighted_compressed_and_clipped = (step(array_adder_object,average_spectrum_phase_compressed_and_clipped , weights_vec_compressed_and_clipped(counter)*abs(fft_vec_phase_compressed_and_clipped).^2));

                    %Find Spectrum Energies over different BW's away from Carrier and Signal Energy over time:
                    [current_fft_energy_in_frequency_bins_compressed_and_clipped] = fft_calculate_energy_between_frequencies(fft_vec_raw_compressed_and_clipped,2,2,1,Fs,frequency_bins);
                    fft_energy_in_frequency_bins_compressed_and_clipped(:,counter) = current_fft_energy_in_frequency_bins_compressed_and_clipped;
                    total_PSD_energy_compressed_and_clipped(counter) = sum(abs(fft_vec_raw_compressed_and_clipped).^2);

                    %Find Peak to Peak and RMS over time:
                    p2p_compressed_and_clipped_vec(counter) = step(p2p_object,current_frame_compressed_and_clipped);
                    rms_compressed_and_clipped_vec(counter) = step(rms_object,current_frame_compressed_and_clipped);
                end
            end %end of if {flag_analyze_during_run_and_after==1}
        end %end of if {flag_only_calculate_raw_statistics==1}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %Plot Results During Loop Run:
        if flag_show_FFT_stats_during_run==1 || flag_show_demodulated_PM_and_FM_phase_during_run==1 || flag_show_raw_vs_compressed_and_clipped_signal_during_run==1

            %SHOW CURRENT FRAME, CURRENT FRAME FT AND PEAK_FOURIER_VALUE OVER TIME:
            if flag_show_FFT_stats_during_run==1
                figure(1);
                subplot(3,1,1)
                plot(current_frame);
                title('current frame'); 
                subplot(3,1,2)
                plot(linspace(-Fs/2,Fs/2,length(fft_vec_raw)),abs(fft_vec_raw));
                ylim([0,4000]);
                title('current frame FFT');
                subplot(3,1,3)
                plot(0:samples_per_frame/Fs:samples_per_frame/Fs*(counter-1),peak_fourier_value_over_time_vec(1:counter));
                xlabel('seconds');
                frame_time = samples_per_frame/Fs*(counter-2);
                title({'peak fourier component in frame', strcat('frame time = ',frame_time)});
                
                if flag_try_compressed_and_clipped_carrier==1
                    figure(2);
                    subplot(3,1,1)
                    plot(current_frame_compressed_and_clipped);
                    title('current frame'); 
                    subplot(3,1,2)
                    plot(linspace(-Fs/2,Fs/2,length(fft_vec_raw_compressed_and_clipped)),abs(fft_vec_raw_compressed_and_clipped));
                    ylim([0,4000]);
                    title('current frame FFT');
                    subplot(3,1,3)
                    plot(0:samples_per_frame/Fs:samples_per_frame/Fs*(counter-1),peak_fourier_value_compressed_and_clipped_vec(1:counter));
                    xlabel('seconds');
                    frame_time = samples_per_frame/Fs*(counter-2);
                    title({'peak fourier component in frame', strcat('frame time = ',frame_time)}); 
                end
            end


            %SHOW PHASE:
            if flag_show_demodulated_PM_and_FM_phase_during_run==1
                figure(3);
                subplot(2,2,1)
                plot(cumsum(phase_signal_corrected));
                title('uncorrected PM');
                subplot(2,2,2)
                plot(phase_signal_corrected);
                title('uncorrected FM');
                subplot(2,2,3)
                plot(filtered_phase);
                title('corrected filtered PM');
                subplot(2,2,4)
                plot(filtered_phase(2:end)-filtered_phase(1:end-1));
                title('corrected filtered FM');
                
                if flag_try_compressed_and_clipped_carrier==1
                    figure(4);
                    subplot(2,2,1)
                    plot(cumsum(phase_signal_corrected));
                    title('uncorrected PM');
                    subplot(2,2,2)
                    plot(phase_signal_corrected);
                    title('uncorrected FM');
                    subplot(2,2,3)
                    plot(filtered_phase);
                    title('corrected filtered PM');
                    subplot(2,2,4)
                    plot(filtered_phase(2:end)-filtered_phase(1:end-1));
                    title('corrected filtered FM');  
                end
                
            end
            
            %SHOW COMPRESSED AND CLIPPED SIGNAL:
            if flag_show_raw_vs_compressed_and_clipped_signal_during_run==1 && flag_try_compressed_and_clipped_carrier==1
                figure(5);
                subplot(2,1,1)
                plot(current_frame,'b');
                title('current raw frame');
                subplot(2,1,2)
                plot(current_frame_compressed_and_clipped);
                title('compressed and clipped current carrier');
            end
            
            drawnow;
        end


        counter=counter+1;
%         toc
%         catch
%            break; 
%         end
    end %end of while loop
end %end of if {flag_should_i_do_any_demodulation==1}
try
    fclose(fid);
    fclose(fid_amplitude);
    fclose(fid_phase);
    release(audio_player_object);
    release(audio_file_writer_demodulated);
    release(logger_object);
    release(audio_file_reader);
    release(audio_file_writer_raw);
    release(logger_object_compressed_and_clipped);
    release(compressed_and_clipped_analytic_signal_object);
    release(analytic_signal_object);
    release(phase_unwrapping_object);
    release(phase_unwrapping_object2);
    release(audio_file_writer_demodulated_compressed_and_clipped);
    fclose('all');
catch 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%sound .wav file generated:
if flag_sound_demodulation_later==1
    %initialize audio_file_reader if i want to sound it immediately after main loop:
    audio_file_reader = dsp.AudioFileReader;
    if flag_sound_raw_signal_or_compressed_and_clipped_signal==1
        audio_file_reader.Filename = audio_file_writer_demodulated.Filename;
        play_wav_file_using_dsp_objects(audio_file_writer_demodulated.Filename,Fs/down_sample_factor,1,0.5);
    elseif flag_sound_raw_signal_or_compressed_and_clipped_signal==2
        audio_file_reader.Filename = audio_file_writer_demodulated_compressed_and_clipped.Filename;
        play_wav_file_using_dsp_objects(audio_file_writer_demodulated_compressed_and_clipped.Filename,Fs/down_sample_factor,1,0.5);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55

if flag_only_calculate_simple_raw_statistics==1
    average_raw_rms = rms(raw_rms_vec);
    average_filtered_frame_rms_vec = rms(filtered_frame_rms_vec);
    average_fft_noise_floor_mean_over_time_vec = mean(fft_noise_floor_mean_over_time_vec);
    raw_rms_vec_different_files(tt) =average_raw_rms;
    filtered_rms_vec_different_files(tt) =average_filtered_frame_rms_vec;
    mean_fft_noise_floor_vec_different_files(tt) =average_fft_noise_floor_mean_over_time_vec;
end


if flag_analyze_during_run_and_after==1 && flag_only_calculate_simple_raw_statistics==0
    %DRAW RESULTS GRAPHS:
    %calculate spectrum of fourier peaks:
    Fc = average_max_frequency;
    t_vec_peaks = 0:samples_per_frame/Fs:samples_per_frame/Fs*(length(peak_fourier_value_over_time_vec)-1);
    t_vec_peaks_mirrored = mirror_array_append(t_vec_peaks,1);
    Fs_peaks = 1/(t_vec_peaks(2)-t_vec_peaks(1));
    [f_vec_average_spectrum] = fft_get_frequency_vec(samples_per_frame,Fs,1);
    [f_vec_phase] = fft_get_frequency_vec(samples_per_frame-1,Fs,1);
    
    %Take only relevant values from phase rms vecs:
    PM_phase_raw_rms_over_time_vec = PM_phase_raw_rms_over_time_vec(2:counter);
    FM_phase_raw_rms_over_time_vec = FM_phase_raw_rms_over_time_vec(2:counter);
    PM_phase_raw_rms_over_time_after_mask_vec = PM_phase_raw_rms_over_time_after_mask_vec(2:counter);
    FM_phase_raw_rms_over_time_after_mask_vec = FM_phase_raw_rms_over_time_after_mask_vec(2:counter);
    PM_phase_filtered_rms_over_time_vec = PM_phase_filtered_rms_over_time_vec(2:counter);
    FM_phase_filtered_rms_over_time_vec = FM_phase_filtered_rms_over_time_vec(2:counter);
    PM_phase_filtered_rms_over_time_after_mask_vec = PM_phase_filtered_rms_over_time_after_mask_vec(2:counter);
    FM_phase_filtered_rms_over_time_after_mask_vec = FM_phase_filtered_rms_over_time_after_mask_vec(2:counter);

    
    %Normalize averaged spectrums by number of frames to compare it to others:
    average_spectrum_raw_signal = average_spectrum_raw_signal./interpolated_transfer_function/counter;
    average_spectrum_amplitude = average_spectrum_amplitude./interpolated_transfer_function/counter;
    average_spectrum_phase_FM = average_spectrum_phase_FM/counter;
    average_spectrum_phase_PM = average_spectrum_phase_PM/counter;

    %normalize weighted spectrum by the sum of weights:
    weights_vec=weights_vec(1:end-1);
    average_spectrum_raw_signal_weighted_original = average_spectrum_raw_signal_weighted./interpolated_transfer_function;
    average_spectrum_raw_signal_weighted = average_spectrum_raw_signal_weighted_original/sum(weights_vec);
    average_spectrum_amplitude_weighted = average_spectrum_amplitude_weighted/sum(weights_vec);
    average_spectrum_phase_FM_weighted = average_spectrum_phase_FM_weighted/sum(weights_vec);
    average_spectrum_phase_PM_weighted = average_spectrum_phase_PM_weighted/sum(weights_vec);
    
    %Calculate fft_peak, CNR, raw RMS, phase RMS over time spectrum:
    CNR_energy_division_over_time_linear_vec = 10.^(CNR_energy_division_over_time_log_vec/10);
    [fft_peaks_over_time_spectrum,f_vec_peaks] = fft_calculate_simple_fft(remove_mean(peak_fourier_value_over_time_vec),Fs_peaks,2,1);
    [CNR_over_time_spectrum,f_vec_peaks] = fft_calculate_simple_fft(remove_mean(CNR_energy_division_over_time_log_vec),Fs_peaks,2,1);
    [raw_rms_over_time_spectrum,f_vec_peaks] = fft_calculate_simple_fft(remove_mean(raw_rms_over_time_vec),Fs_peaks,2,1);
    [FM_phase_filtered_rms_over_time_spectrum,f_vec_peaks] = fft_calculate_simple_fft(remove_mean(FM_phase_filtered_rms_over_time_vec),Fs_peaks,2,1);
    [PM_phase_filtered_rms_over_time_spectrum,f_vec_peaks] = fft_calculate_simple_fft(remove_mean(PM_phase_filtered_rms_over_time_vec),Fs_peaks,2,1);
    [FM_phase_raw_rms_over_time_spectrum,f_vec_peaks] = fft_calculate_simple_fft(remove_mean(FM_phase_raw_rms_over_time_vec),Fs_peaks,2,1);
    [PM_phase_raw_rms_over_time_spectrum,f_vec_peaks] = fft_calculate_simple_fft(remove_mean(PM_phase_raw_rms_over_time_vec),Fs_peaks,2,1);
    
    %Get mean and std log and linear CNR:
    %Up until now the mean CNR was taken as the CNR calculated from the averaged spectrum:
    mean_linear_CNR_energy_division = mean(CNR_energy_division_over_time_linear_vec);
    mean_log_CNR_energy_division = mean(CNR_energy_division_over_time_log_vec);
    log_of_mean_linear_CNR_energy_division = 10*log10(mean_linear_CNR_energy_division);
    std_linear_CNR_energy_division = std(CNR_energy_division_over_time_linear_vec);
    std_log_CNR_energy_division = std(CNR_energy_division_over_time_log_vec);
    CNR_above_noise_floor_over_time_linear_vec = 10.^(CNR_above_noise_floor_over_time_log_vec/10);
      
    %TAKE OUT 2 LAST VALUES BECAUSE THEY WERE NOT ASSIGNED DURING RUN AND RUIN REGRESSION:
    CNR_energy_division_over_time_linear_new = CNR_energy_division_over_time_linear_vec(1:end-2);
    CNR_over_time_log_new = CNR_energy_division_over_time_log_vec(1:end-2);
    peak_fourier_value_over_time_new = peak_fourier_value_over_time_vec(1:end-2);
    raw_rms_over_time_new = raw_rms_over_time_vec(1:end-2);
    PSD_total_energy_over_time_new = PSD_total_energy_over_time_vec(1:end-2);
    carrier_lobe_energy_over_time_new = carrier_lobe_energy_over_time_vec(1:end-2);
    FM_phase_filtered_rms_over_time_new = FM_phase_filtered_rms_over_time_vec(1:end-2);
    PM_phase_filtered_rms_over_time_new = PM_phase_filtered_rms_over_time_vec(1:end-2); 
    FM_phase_raw_rms_over_time_new = FM_phase_raw_rms_over_time_vec(1:end-2);
    PM_phase_raw_rms_over_time_new = PM_phase_raw_rms_over_time_vec(1:end-2); 
    
       
%     %FIT LORENTZIAN TO AVERAGE RAW SPECTRUM:
%     BW_fit = 500;
%     start_frequency_fit = Fc-BW_fit;
%     stop_frequency_fit = Fc+BW_fit;
%     [f_vec_average_spectrum_two_sided] = fft_get_frequency_vec(samples_per_frame,Fs,0);
%     f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(average_spectrum_raw_signal),average_max_frequency,BW_fit/5]);
%     ft = fittype('A/(1+(x-b).^2/sigma^2)','options',f0);
%     f_vec_elements_to_fit = fft_get_certain_elements(f_vec_average_spectrum_two_sided(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
%     spectrum_elements_to_fit =  fft_get_certain_elements(average_spectrum_raw_signal(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
%     [fitted_parameters_average_spectrum,bla1] = fit(f_vec_elements_to_fit,spectrum_elements_to_fit,ft);
%     average_carrier_linear_lorentzian_fit_sigma = fitted_parameters_average_spectrum.sigma;
%     
%     %PLOT LORENTZIAN FIT TO AVERAGE RAW SPECTRUM:
%     figure;
%     scatter(f_vec_elements_to_fit,spectrum_elements_to_fit);
%     hold on;
%     plot(fitted_parameters_average_spectrum,'r');
%     title({'Lorentzian fit to average raw spectrum',texlabel(fitted_parameters_average_spectrum),strcat('sigma=',num2str(fitted_parameters_average_spectrum.sigma),'[Hz]'),strcat('Frame Time = ',num2str(time_for_FFT_frame))});
%     xlabel('Frequency[Hz]');
%     ylabel('PSD [linear]');
%     axes_set_all_labels_font_size(15);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' average raw spectrum fit.jpg'),'jpg'); 
%     end
%     
%     %FIT 1/F TO AVERAGE FM PHASE SPECTRUM LINEAR:
%     BW_fit = 2500;
%     start_frequency_fit = 0-BW_fit; 
%     stop_frequency_fit = 0+BW_fit;
%     average_spectrum_phase_FM_for_fit = (average_spectrum_phase_FM);
%     [f_vec_average_spectrum_two_sided] = fft_get_frequency_vec(length(average_spectrum_phase_FM_for_fit),Fs,0);
%     f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(average_spectrum_phase_FM_for_fit),0,BW_fit/20]);
%     ft = fittype('A/(1+abs((x-b))/sigma)','options',f0);
%     f_vec_elements_to_fit = fft_get_certain_elements(f_vec_average_spectrum_two_sided(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
%     spectrum_elements_to_fit =  fft_get_certain_elements((average_spectrum_phase_FM_for_fit(:)),Fs,3,2,start_frequency_fit,stop_frequency_fit);
%     [fitted_parameters_average_spectrum,bla1] = fit(f_vec_elements_to_fit,spectrum_elements_to_fit,ft);
%     average_FM_phase_linear_lorentzian_fit_sigma = fitted_parameters_average_spectrum.sigma;
%     
%     %PLOT 1/F FIT TO AVERAGE FM PHASE SPECTRUM LINEAR:
%     figure;
%     scatter(f_vec_elements_to_fit,spectrum_elements_to_fit);
%     hold on;
%     plot(f_vec_elements_to_fit,(feval(fitted_parameters_average_spectrum,f_vec_elements_to_fit)),'r');
%     title({'Lorentzian fit to average FM phase spectrum linear',texlabel(fitted_parameters_average_spectrum),strcat('sigma=',num2str(abs(fitted_parameters_average_spectrum.sigma)),'[Hz]'),strcat('Frame Time = ',num2str(time_for_FFT_frame))});
%     xlabel('Frequency[Hz]');
%     ylabel('PSD [linear]');
%     axes_set_all_labels_font_size(15);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' average FM phase linear spectrum fit.jpg'),'jpg'); 
%     end
% 
%     %FIT LORENTZIAN TO AVERAGE PM PHASE SPECTRUM LINEAR:
%     BW_fit = 1500;
%     start_frequency_fit = 0-BW_fit;
%     stop_frequency_fit = 0+BW_fit;
%     average_spectrum_phase_PM_for_fit = (average_spectrum_phase_PM);
%     [f_vec_average_spectrum_two_sided] = fft_get_frequency_vec(length(average_spectrum_phase_PM_for_fit),Fs,0);
%     f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(average_spectrum_phase_PM_for_fit),0,BW_fit/30]);
%     ft = fittype('A/(1+((x-b).^2)/sigma.^2)','options',f0);
%     f_vec_elements_to_fit = fft_get_certain_elements(f_vec_average_spectrum_two_sided(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
%     spectrum_elements_to_fit =  fft_get_certain_elements(average_spectrum_phase_PM_for_fit(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
%     [fitted_parameters_average_spectrum,bla1] = fit(f_vec_elements_to_fit,spectrum_elements_to_fit,ft);
%     average_PM_phase_linear_lorentzian_fit_sigma = fitted_parameters_average_spectrum.sigma;
%     
%     %PLOT LORENTZIAN FIT TO AVERAGE PM PHASE SPECTRUM LINEAR:
%     figure;
%     scatter(f_vec_elements_to_fit,spectrum_elements_to_fit);
%     hold on;
%     plot(fitted_parameters_average_spectrum,'r');
%     title({'Lorentzian fit to average PM phase spectrum linear',texlabel(fitted_parameters_average_spectrum),strcat('sigma=',num2str(abs(fitted_parameters_average_spectrum.sigma)),'[Hz]'),strcat('Frame Time = ',num2str(time_for_FFT_frame))});
%     xlabel('Frequency[Hz]');
%     ylabel('PSD [linear]');
%     axes_set_all_labels_font_size(15);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' average PM phase linear spectrum fit.jpg'),'jpg'); 
%     end
% 
%     %FIT 1/F TO AVERAGE FM PHASE SPECTRUM LOGARITHMIC:
%     BW_fit = 500;
%     start_frequency_fit = 0-BW_fit; 
%     stop_frequency_fit = 0+BW_fit;
%     average_spectrum_phase_FM_for_fit = 10*log10(average_spectrum_phase_FM);
%     [f_vec_average_spectrum_two_sided] = fft_get_frequency_vec(length(average_spectrum_phase_FM_for_fit),Fs,0);
%     f_vec_elements_to_fit = fft_get_certain_elements(f_vec_average_spectrum_two_sided(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
%     spectrum_elements_to_fit =  fft_get_certain_elements((average_spectrum_phase_FM_for_fit(:)),Fs,3,2,start_frequency_fit,stop_frequency_fit);
%     f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[min(spectrum_elements_to_fit),max(spectrum_elements_to_fit),0,BW_fit]);
%     ft = fittype('B+A/(1+abs((x-b))/sigma)','options',f0);
%     [fitted_parameters_average_spectrum,bla1] = fit(f_vec_elements_to_fit,spectrum_elements_to_fit,ft);
%     average_FM_phase_log_lorentzian_fit_sigma = fitted_parameters_average_spectrum.sigma;
%     
%     %PLOT 1/F FIT TO AVERAGE PM PHASE SPECTRUM LOGARITHMIC:
%     figure;
%     scatter(f_vec_elements_to_fit,spectrum_elements_to_fit);
%     hold on;
%     plot(f_vec_elements_to_fit,(feval(fitted_parameters_average_spectrum,f_vec_elements_to_fit)),'r');
%     title({'Lorentzian fit to average FM phase spectrum logarithmic',texlabel(fitted_parameters_average_spectrum),strcat('sigma=',num2str(abs(fitted_parameters_average_spectrum.sigma)),'[Hz]'),strcat('Frame Time = ',num2str(time_for_FFT_frame))});
%     xlabel('Frequency[Hz]');
%     ylabel('PSD [dB]');
%     axes_set_all_labels_font_size(15);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' average PM phase logarithmic spectrum fit.jpg'),'jpg'); 
%     end

    %FIT 1/F TO AVERAGE PM PHASE SPECTRUM LOGARITHMIC:
    BW_fit = 2500;
    start_frequency_fit = 0-BW_fit;
    stop_frequency_fit = 0+BW_fit;
    average_spectrum_phase_PM_for_fit = 10*log10(average_spectrum_phase_PM);
    [f_vec_average_spectrum_two_sided] = fft_get_frequency_vec(length(average_spectrum_phase_PM_for_fit),Fs,0);
    f_vec_elements_to_fit = fft_get_certain_elements(f_vec_average_spectrum_two_sided(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
    spectrum_elements_to_fit =  fft_get_certain_elements(average_spectrum_phase_PM_for_fit(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
    f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[min(spectrum_elements_to_fit),max(spectrum_elements_to_fit),0,BW_fit]);
    ft = fittype('B+A/(1+abs((x-b))/sigma)','options',f0);
    [fitted_parameters_average_spectrum,bla1] = fit(f_vec_elements_to_fit,spectrum_elements_to_fit,ft);
    average_PM_phase_log_lorentzian_fit_sigma = fitted_parameters_average_spectrum.sigma;
    
%     %PLOT 1/F FIT TO AVERAGE PM PHASE SPECTRUM LOGARITHMIC:
%     figure;
%     scatter(f_vec_elements_to_fit,spectrum_elements_to_fit);
%     hold on;
%     plot(fitted_parameters_average_spectrum,'r');
%     title({'Lorentzian fit to average PM phase spectrum logarithmic',texlabel(fitted_parameters_average_spectrum),strcat('sigma=',num2str((fitted_parameters_average_spectrum.sigma)),'[Hz]'),strcat('Frame Time = ',num2str(time_for_FFT_frame))});
%     xlabel('Frequency[Hz]');
%     ylabel('PSD [dB]');
%     axes_set_all_labels_font_size(15);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' average PM phase logarithmic spectrum fit.jpg'),'jpg'); 
%     end

    %FIT LORENTZIAN TO AVERAGE ANALYTICAL AMPLITUDE SPECTRUM:
    BW_fit = 1500;
    start_frequency_fit = 0-BW_fit;
    stop_frequency_fit = 0+BW_fit; 
    average_spectrum_amplitude_for_fit = 10*log10(average_spectrum_amplitude);
    [f_vec_average_spectrum_two_sided] = fft_get_frequency_vec(length(average_spectrum_amplitude_for_fit),Fs,0);
    f_vec_elements_to_fit = fft_get_certain_elements(f_vec_average_spectrum_two_sided(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
    spectrum_elements_to_fit =  fft_get_certain_elements(average_spectrum_amplitude_for_fit(:),Fs,3,2,start_frequency_fit,stop_frequency_fit);
    f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[min(spectrum_elements_to_fit),max(spectrum_elements_to_fit),0,BW_fit/4]);
    ft = fittype('B+A/(1+abs((x-b))/sigma)','options',f0);
    [fitted_parameters_average_spectrum,bla1] = fit(f_vec_elements_to_fit,spectrum_elements_to_fit,ft);
    average_analytical_amplitude_linear_lorentzian_fit_sigma = fitted_parameters_average_spectrum.sigma;
    
%     %PLOT LORENTZIAN FIT TO AVERAGE ANALYTICAL AMPLITUDE SPECTRUM:
%     figure;
%     scatter(f_vec_elements_to_fit,spectrum_elements_to_fit);
%     hold on;
%     plot(fitted_parameters_average_spectrum,'r');
%     title({'Lorentzian fit to average analytic amplitude spectrum',texlabel(fitted_parameters_average_spectrum),strcat('sigma=',num2str(fitted_parameters_average_spectrum.sigma),'[Hz]'),strcat('Frame Time = ',num2str(time_for_FFT_frame)),'A Measure for AM'});
%     xlabel('Frequency[Hz]');
%     ylabel('PSD [linear]');
%     axes_set_all_labels_font_size(15);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' average analytical amplitude logarithmic spectrum fit.jpg'),'jpg'); 
%     end
% 
%     %CALCULATE RAW STATISTICS AUDO-CORRELATIONS:
%     %Calculate fft_peak, CNR, raw RMS, phase RMS over time auto-correlation:
%     auto_correlation_fft_peaks = autocorr(remove_mean(peak_fourier_value_over_time_new),length(peak_fourier_value_over_time_new)-1);
%     auto_correlation_log_CNR = autocorr(remove_mean(CNR_over_time_log_new),length(peak_fourier_value_over_time_new)-1);
%     auto_correlation_linear_CNR = autocorr(remove_mean(CNR_energy_division_over_time_linear_new),length(peak_fourier_value_over_time_new)-1);
%     auto_correlation_raw_rms = autocorr(remove_mean(raw_rms_over_time_new),length(peak_fourier_value_over_time_new)-1);
%     auto_correlation_FM_phase_filtered = autocorr(remove_mean(FM_phase_filtered_rms_over_time_new),length(peak_fourier_value_over_time_new)-1);
%     auto_correlation_PM_phase_filtered = autocorr(remove_mean(PM_phase_filtered_rms_over_time_new),length(peak_fourier_value_over_time_new)-1);
%     auto_correlation_FM_phase_raw = autocorr(remove_mean(FM_phase_raw_rms_over_time_new),length(FM_phase_raw_rms_over_time_new)-1);
%     auto_correlation_PM_phase_raw = autocorr(remove_mean(PM_phase_raw_rms_over_time_new),length(PM_phase_raw_rms_over_time_new)-1);
%     auto_correlation_PSD_total_energy = autocorr(remove_mean(PSD_total_energy_over_time_new),length(PSD_total_energy_over_time_new)-1);
%     auto_correlation_carrier_lobe_energy = autocorr(remove_mean(carrier_lobe_energy_over_time_new),length(carrier_lobe_energy_over_time_new)-1);
%     
%     auto_correlation_fft_peaks = mirror_array_append(auto_correlation_fft_peaks,0);
%     auto_correlation_log_CNR = mirror_array_append(auto_correlation_log_CNR,0);
%     auto_correlation_linear_CNR = mirror_array_append(auto_correlation_linear_CNR,0);
%     auto_correlation_raw_rms = mirror_array_append(auto_correlation_raw_rms,0);
%     auto_correlation_FM_phase_filtered = mirror_array_append(auto_correlation_FM_phase_filtered,0);
%     auto_correlation_PM_phase_filtered = mirror_array_append(auto_correlation_PM_phase_filtered,0);
%     auto_correlation_FM_phase_raw = mirror_array_append(auto_correlation_FM_phase_raw,0);
%     auto_correlation_PM_phase_raw = mirror_array_append(auto_correlation_PM_phase_raw,0); 
%     auto_correlation_PSD_total_energy = mirror_array_append(auto_correlation_PSD_total_energy,0);
% 
%     zoom_radius=10;
%     [zoomed_CNR_linear_auto_correlation,start_index_CNR,stop_index_CNR] = zoom_around_max(auto_correlation_linear_CNR,zoom_radius);
%     [zoomed_raw_rms_auto_correlation,start_index_raw_rms,stop_index_raw_rms] = zoom_around_max(auto_correlation_raw_rms,zoom_radius);
%     [zoomed_peak_fourier_value_auto_correlation,start_index_peak_fourier,stop_index_peak_fourier] = zoom_around_max(auto_correlation_fft_peaks,zoom_radius);
%     [zoomed_PSD_total_energy_auto_correlation,start_index_PSD_total_energy,stop_index_PSD_total_energy] = zoom_around_max(auto_correlation_PSD_total_energy,zoom_radius);
%     [zoomed_carrier_lobe_energy_auto_correlation,start_index_carrier_lobe_energy,stop_index_carrier_lobe_energy] = zoom_around_max(auto_correlation_carrier_lobe_energy,zoom_radius);
%     [zoomed_FM_raw_phase_rms_auto_correlation,start_index_FM_raw_phase_rms,stop_index_FM_raw_phase_rms] = zoom_around_max(auto_correlation_FM_phase_raw,zoom_radius);
%     [zoomed_PM_raw_phase_rms_auto_correlation,start_index_PM_raw_phase_rms,stop_index_PM_raw_phase_rms] = zoom_around_max(auto_correlation_PM_phase_raw,zoom_radius);
%     [zoomed_FM_filtered_phase_rms_auto_correlation,start_index_FM_filtered_phase_rms,stop_index_FM_filtered_phase_rms] = zoom_around_max(auto_correlation_FM_phase_filtered,zoom_radius);
%     [zoomed_PM_filtered_phase_rms_auto_correlation,start_index_PM_filtered_phase_rms,stop_index_PM_filtered_phase_rms] = zoom_around_max(auto_correlation_PM_phase_filtered,zoom_radius);
%     
%     %PLOT AUTO CORRELATIONS:
%     x_vec = linspace(-length(CNR_energy_division_over_time_linear_new),length(CNR_energy_division_over_time_linear_new),2*length(CNR_energy_division_over_time_linear_new)-1);
%     figure;
%     subplot(2,3,1);
%     plot(x_vec(start_index_CNR:stop_index_CNR),zoomed_CNR_linear_auto_correlation);
%     title('CNR auto-correlation');
%     ylim([min(zoomed_CNR_linear_auto_correlation),max(zoomed_CNR_linear_auto_correlation)]);
%     subplot(2,3,2);
%     plot(x_vec(start_index_raw_rms:stop_index_raw_rms),zoomed_raw_rms_auto_correlation);
%     title('raw rms auto-correlation');
%     ylim([min(zoomed_raw_rms_auto_correlation),max(zoomed_raw_rms_auto_correlation)]);
%     subplot(2,3,3);
%     plot(x_vec(start_index_peak_fourier:stop_index_peak_fourier),zoomed_peak_fourier_value_auto_correlation);
%     title('peak fourier value auto-correlation');
%     ylim([min(zoomed_peak_fourier_value_auto_correlation),max(zoomed_peak_fourier_value_auto_correlation)]);
%     subplot(2,3,4);
%     plot(x_vec(start_index_PSD_total_energy:stop_index_PSD_total_energy),zoomed_PSD_total_energy_auto_correlation);
%     title('PSD total energy auto-correlation');
%     ylim([min(zoomed_PSD_total_energy_auto_correlation),max(zoomed_PSD_total_energy_auto_correlation)]);
%     subplot(2,3,5);
%     plot(x_vec(start_index_carrier_lobe_energy:stop_index_carrier_lobe_energy),zoomed_carrier_lobe_energy_auto_correlation);
%     title('carrier lobe energy auto-correlation');
%     ylim([min(zoomed_carrier_lobe_energy_auto_correlation),max(zoomed_carrier_lobe_energy_auto_correlation)]);
%     subplot(2,3,6);
%     plot(x_vec(start_index_FM_raw_phase_rms:stop_index_FM_raw_phase_rms),zoomed_FM_raw_phase_rms_auto_correlation);
%     title('FM raw phase rms auto-correlation');
%     ylim([min(zoomed_FM_raw_phase_rms_auto_correlation),max(zoomed_FM_raw_phase_rms_auto_correlation)]);
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' different auto correlations.jpg'),'jpg'); 
%     end
%     
%     %RAW STATISTICS CROSS-CORRELATIONS:
%     zoom_radius=10;
%     [zoomed_CNR_FM_crosscorr,start_index_CNR_FM,stop_index_CNR_FM] = zoom_around_max(xcov(CNR_energy_division_over_time_linear_new,FM_phase_filtered_rms_over_time_new,'coeff'),zoom_radius);
%     [zoomed_raw_rms_FM_crosscorr,start_index_raw_rms_FM,stop_index_raw_rms_FM] = zoom_around_max(xcov(raw_rms_over_time_new,FM_phase_filtered_rms_over_time_new,'coeff'),zoom_radius);
%     [zoomed_CNR_raw_rms_crosscorr,start_index_CNR_raw_rms,stop_index_CNR_raw_rms] = zoom_around_max(xcov(CNR_energy_division_over_time_linear_new,raw_rms_over_time_new,'coeff'),zoom_radius);
%     [zoomed_peak_fourier_PSD_crosscorr,start_index_peak_fourier_PSD,stop_index_peak_fourier_PSD] = zoom_around_max(xcov(peak_fourier_value_over_time_new,PSD_total_energy_over_time_new,'coeff'),zoom_radius);
%     [zoomed_carrier_lobe_PSD_crosscorr,start_index_carrier_lobe_PSD,stop_index_carrier_lobe_PSD] = zoom_around_max(xcov(carrier_lobe_energy_over_time_new,PSD_total_energy_over_time_new,'coeff'),zoom_radius);
%     [zoomed_CNR_PSD_crosscorr,start_index_CNR_PSD,stop_index_CNR_PSD] = zoom_around_max(xcov(CNR_energy_division_over_time_linear_new,PSD_total_energy_over_time_new,'coeff'),zoom_radius);
%     [zoomed_peak_fourier_CNR_crosscorr,start_index_peak_fourier_CNR,stop_index_peak_fourier_CNR] = zoom_around_max(xcov(peak_fourier_value_over_time_new,CNR_energy_division_over_time_linear_new,'coeff'),zoom_radius);
% 
%     x_vec = linspace(-length(CNR_energy_division_over_time_linear_new),length(CNR_energy_division_over_time_linear_new),2*length(CNR_energy_division_over_time_linear_new)-1);
%     x_vec_CNR_FM = x_vec(start_index_CNR_FM:stop_index_CNR_FM);
%     x_vec_raw_rms_FM = x_vec(start_index_raw_rms_FM:stop_index_raw_rms_FM);
%     x_vec_CNR_raw_rms = x_vec(start_index_CNR_raw_rms:stop_index_CNR_raw_rms);
%     x_vec_peak_fourier_PSD = x_vec(start_index_peak_fourier_PSD:stop_index_peak_fourier_PSD);
%     x_vec_carrier_lobe_PSD = x_vec(start_index_carrier_lobe_PSD:stop_index_carrier_lobe_PSD);
%     x_vec_CNR_PSD = x_vec(start_index_CNR_PSD:stop_index_CNR_PSD);
%     x_vec_peak_fourier_CNR = x_vec(start_index_peak_fourier_CNR:stop_index_peak_fourier_CNR);

%     figure;
%     subplot(3,3,1);
%     plot(x_vec_CNR_FM,zoomed_CNR_FM_crosscorr);
%     title('CNR vs. Phase rms');
%     subplot(3,3,2);
%     plot(x_vec_raw_rms_FM,zoomed_raw_rms_FM_crosscorr);
%     title('raw rms vs. Phase rms');
%     ylim([min(zoomed_raw_rms_FM_crosscorr),max(zoomed_raw_rms_FM_crosscorr)]);
%     subplot(3,3,3);
%     plot(x_vec_CNR_raw_rms,zoomed_CNR_raw_rms_crosscorr);
%     title('CNR vs. raw rms');
%     ylim([min(zoomed_CNR_raw_rms_crosscorr),max(zoomed_CNR_raw_rms_crosscorr)]);
%     subplot(3,3,4);
%     plot(x_vec_peak_fourier_PSD,zoomed_peak_fourier_PSD_crosscorr);
%     title('fourier peak vs. PSD total energy');
%     ylim([min(zoomed_peak_fourier_PSD_crosscorr),max(zoomed_peak_fourier_PSD_crosscorr)]);
%     subplot(3,3,5);
%     plot(x_vec_carrier_lobe_PSD,zoomed_carrier_lobe_PSD_crosscorr);
%     title('carrier lobe energy vs. PSD total energy');
%     ylim([min(zoomed_carrier_lobe_PSD_crosscorr),max(zoomed_carrier_lobe_PSD_crosscorr)]);
%     subplot(3,3,6);
%     plot(x_vec_CNR_PSD,zoomed_CNR_PSD_crosscorr);
%     title('CNR vs. PSD total energy');
%     ylim([min(zoomed_CNR_PSD_crosscorr),max(zoomed_CNR_PSD_crosscorr)]);
%     subplot(3,3,7);
%     plot(x_vec_peak_fourier_CNR,zoomed_peak_fourier_CNR_crosscorr);
%     title('peak fourier value vs. CNR');
%     ylim([min(zoomed_peak_fourier_CNR_crosscorr),max(zoomed_peak_fourier_CNR_crosscorr)]);
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' different cross correlations.jpg'),'jpg'); 
%     end
%     
% %     %RAW STATISTICS REGRESSIONS:
% %     figure;
% %     subplot(3,3,1);
% %     [new_x,new_y,plot_handle] = regression_plot(CNR_energy_division_over_time_linear_new,FM_phase_filtered_rms_over_time_new,9);
% %     title('CNR vs. Phase rms');
% %     subplot(3,3,2);
% %     [new_x,new_y,plot_handle] = regression_plot(raw_rms_over_time_new,FM_phase_filtered_rms_over_time_new,9);
% %     title('raw rms vs. Phase rms');
% %     subplot(3,3,3);
% %     [new_x,new_y,plot_handle] = regression_plot(CNR_energy_division_over_time_linear_new,raw_rms_over_time_new,9);
% %     title('CNR vs. raw rms');
% %     subplot(3,3,4);
% %     [new_x,new_y,plot_handle] = regression_plot(peak_fourier_value_over_time_new,PSD_total_energy_over_time_new,9);
% %     title('fourier peak vs. PSD total energy');
% %     subplot(3,3,5);
% %     [new_x,new_y,plot_handle] = regression_plot(carrier_lobe_energy_over_time_new,PSD_total_energy_over_time_new,9);
% %     title('carrier lobe energy vs. PSD total energy');
% %     subplot(3,3,6);
% %     [new_x,new_y,plot_handle] = regression_plot(CNR_energy_division_over_time_linear_new,PSD_total_energy_over_time_new,9);
% %     title('CNR vs. PSD total energy');
% %     subplot(3,3,7);
% %     [new_x,new_y,plot_handle] = regression_plot(peak_fourier_value_over_time_new,CNR_energy_division_over_time_linear_new,9);
% %     title('peak fourier value vs. CNR');
% %     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
% %     if flag_save_plots==1
% %        saveas(gcf,strcat(fullfile(directory,raw_file_name),' regressions of raw signals.jpg'),'jpg'); 
% %     end
% % 
% %     %DIFFERENCE STATISTICS REGRESSIONS:
% %     figure;
% %     subplot(3,3,1);
% %     [new_x,new_y,plot_handle] = regression_plot(diff(CNR_energy_division_over_time_linear_new),diff(FM_phase_filtered_rms_over_time_new),9);
% %     title('diff CNR vs. diff Phase rms');
% %     subplot(3,3,2);
% %     [new_x,new_y,plot_handle] = regression_plot(diff(raw_rms_over_time_new),diff(FM_phase_filtered_rms_over_time_new),9);
% %     title('diff raw rms vs. diff Phase rms');
% %     subplot(3,3,3);
% %     [new_x,new_y,plot_handle] = regression_plot(diff(CNR_energy_division_over_time_linear_new),diff(raw_rms_over_time_new),9);
% %     title('diff CNR vs. diff raw rms');
% %     subplot(3,3,4);
% %     [new_x,new_y,plot_handle] = regression_plot(diff(peak_fourier_value_over_time_new),diff(PSD_total_energy_over_time_new),9);
% %     title('diff fourier peak vs. diff PSD total energy');
% %     subplot(3,3,5);
% %     [new_x,new_y,plot_handle] = regression_plot(diff(carrier_lobe_energy_over_time_new),diff(PSD_total_energy_over_time_new),9);
% %     title('diff carrier lobe energy vs. diff PSD total energy');
% %     subplot(3,3,6);
% %     [new_x,new_y,plot_handle] = regression_plot(diff(CNR_energy_division_over_time_linear_new),diff(PSD_total_energy_over_time_new),9);
% %     title('diff CNR vs. diff PSD total energy');
% %     subplot(3,3,7);
% %     [new_x,new_y,plot_handle] = regression_plot(diff(peak_fourier_value_over_time_new),diff(CNR_energy_division_over_time_linear_new),9);
% %     title('diff peak fourier value vs. diff CNR');
% %     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
% %     if flag_save_plots==1
% %        saveas(gcf,strcat(fullfile(directory,raw_file_name),' regressions of differential signal.jpg'),'jpg'); 
% %     end
     
    %Fit a gaussian to the carrier lobe and calculate effective CNR:
    %fit gaussian to averaged and weighted lobes:
    f_vec_average_spectrum = f_vec_average_spectrum(2:end);
    f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(average_spectrum_raw_signal),average_max_frequency,200]);
    ft = fittype('A*exp(-(x-b)^2/(2*sigma^2))','options',f0);
    [fitted_parameters_average_spectrum,bla1] = fit(f_vec_average_spectrum(:),average_spectrum_raw_signal(ceil(length(average_spectrum_raw_signal)/2)+(1-mod(length(average_spectrum_raw_signal),2)):end),ft);
    fitted_parameters_average_raw_spectrum_sigma = fitted_parameters_average_spectrum.sigma;
    
    f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(average_spectrum_raw_signal_weighted),average_max_frequency,200]);
    ft = fittype('A*exp(-(x-b)^2/(2*sigma^2))','options',f0);
    [fitted_parameters_average_spectrum_weighted,bla2] = fit(f_vec_average_spectrum(:),average_spectrum_raw_signal_weighted(ceil(length(average_spectrum_raw_signal_weighted)/2)+(1-mod(length(average_spectrum_raw_signal_weighted),2)):end),ft);
    fitted_parameters_average_raw_spectrum_weighted_sigma = fitted_parameters_average_spectrum_weighted.sigma;
    
    f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(max_spectrum),average_max_frequency,200]);
    ft = fittype('A*exp(-(x-b)^2/(2*sigma^2))','options',f0);
    [fitted_parameters_max_spectrum,bla3] = fit(f_vec_average_spectrum(:),max_spectrum(ceil(length(max_spectrum)/2)+(1-mod(length(max_spectrum),2)):end),ft);
    fitted_parameters_max_raw_spectrum_weighted_sigma = fitted_parameters_max_spectrum.sigma;
    
    %Calculate average CNRs of averaged spectrum:
    BW = 5000;
    [~,~,noise_floor_left_start_averaged,~,~,noise_floor_right_stop_averaged] = fft_get_carrier_lobe_and_noise_frequencies(average_max_frequency,effective_lobe_BW_one_sided,BW);  
    %calculate average CNRs of averaged and weighted spectrum:
    [CNR_averaged,CNR_above_noise_floor_with_RBW1Hz_averaged,CNR_above_noise_floor_with_RBW1Hz_averaged_vec,carrier_max_averaged,mean_noise_floor_averaged] = ...
        calculate_CNR(average_spectrum_raw_signal,3,BW,effective_lobe_BW_one_sided,Fs,average_max_frequency);
    [CNR_averaged_weighted,CNR_above_noise_floor_with_RBW1Hz_averaged_weighted,CNR_above_noise_with_RBW1Hz_averaged_weighted_vec,carrier_max_averaged_weighted,mean_noise_floor_averaged_weighted] = ...
        calculate_CNR(average_spectrum_raw_signal_weighted,3,BW,effective_lobe_BW_one_sided,Fs,average_max_frequency); 
    %get relevant PSDs and f_vecs for plotting:
    [zoomed_in_PSD_average,zoomed_in_f_vec_averaged] = get_fft_between_certain_frequencies(average_spectrum_raw_signal,Fs,3,2,noise_floor_left_start_averaged,noise_floor_right_stop_averaged);
    [zoomed_in_PSD_averaged_weighted,zoomed_in_f_vec_averaged_weighted] = get_fft_between_certain_frequencies(average_spectrum_raw_signal_weighted,Fs,3,2,noise_floor_left_start_averaged,noise_floor_right_stop_averaged);
    %get PSD energy as a function of offset from carrier:
    [energy_vec_from_carrier_average,frequency_offset_from_carrier_vec_average] = fft_get_PSD_energy_vs_distance_from_carrier(average_spectrum_raw_signal,Fs,average_max_frequency,BW,3,3,1);
    [energy_vec_from_carrier_average_weighted,frequency_offset_from_carrier_vec_average_weighted] = fft_get_PSD_energy_vs_distance_from_carrier(average_spectrum_raw_signal_weighted,Fs,average_max_frequency,BW,3,3,1);
       
    
%     %plot results for fourier peak over time:
%     figure;
%     subplot(2,2,1)
%     plot(t_vec_peaks,peak_fourier_value_over_time_vec,'b',t_vec_peaks,fft_noise_floor_mean_over_time_vec,'r');
%     legend({'Peak fourier value over time','Noise floor over time'},'location','best','fontsize',10);
%     title('peak fourier component over time');
%     xlabel('Time[Sec]');
%     subplot(2,2,2)
%     plot(f_vec_peaks,abs(fft_peaks_over_time_spectrum));
%     title('spectrum of fourier peaks over time');
%     xlabel('Frequency[Hz]');
%     subplot(2,2,3)
%     plot(t_vec_peaks_mirrored(3:end-2),auto_correlation_fft_peaks);
%     title('fourier peak value over time auto correlation');
%     subplot(2,2,4)
%     hist(peak_fourier_value_over_time_vec,100);
%     title('histogram of carrier fft peak');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' fouirer peak over time properties.jpg'),'jpg'); 
%     end
    
    
%     %plot results for FM raw phase rms over time:
%     figure;
%     subplot(2,2,1)
%     plot(t_vec_peaks,FM_phase_raw_rms_over_time_vec,'b');
%     hold on;
%     plot(t_vec_peaks,mean(FM_phase_raw_rms_over_time_vec)*ones(length(t_vec_peaks),1),'r');
%     legend({'FM raw phase rms over time','mean raw phase rms over whole recording'},'location','best','fontsize',10);
%     title({'FM raw phase rms over time',strcat('Frequencies Filtered: ',num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),'[Hz]')});
%     xlabel('Time[Sec]');
%     subplot(2,2,2) 
%     plot(f_vec_peaks,abs(FM_phase_raw_rms_over_time_spectrum));
%     title('spectrum of FM raw phase rms over time');
%     xlabel('Frequency[Hz]');
%     subplot(2,2,3)
%     plot(t_vec_peaks_mirrored(3:end-2),auto_correlation_FM_phase_raw);
%     title('FM raw rms over time auto corrrelation');
%     subplot(2,2,4)
%     hist(FM_phase_raw_rms_over_time_vec,100);
%     title('histogram of FM raw phase rms');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' FM raw phase rms over time properties.jpg'),'jpg'); 
%     end
    
%     %plot results for FM filtered phase rms over time:
%     figure;
%     subplot(2,2,1)
%     plot(t_vec_peaks,FM_phase_filtered_rms_over_time_vec,'b');
%     hold on;
%     plot(t_vec_peaks,mean(FM_phase_filtered_rms_over_time_vec)*ones(length(t_vec_peaks),1),'r');
%     legend({'FM filtered phase rms over time','mean filtered phase rms over whole recording'},'location','best','fontsize',10);
%     title({'FM filtered phase rms over time',strcat('Frequencies Filtered: ',num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),'[Hz]')});
%     xlabel('Time[Sec]');
%     subplot(2,2,2) 
%     plot(f_vec_peaks,abs(FM_phase_filtered_rms_over_time_spectrum));
%     title('spectrum of FM filtered phase rms over time');
%     xlabel('Frequency[Hz]');
%     subplot(2,2,3)
%     plot(t_vec_peaks_mirrored(3:end-2),auto_correlation_FM_phase_filtered);
%     title('FM filtered phase rms over time auto correlation');
%     subplot(2,2,4)
%     hist(FM_phase_filtered_rms_over_time_vec,100);
%     title('histogram of FM filtered phase rms');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' FM filtered phase rms over time properties.jpg'),'jpg'); 
%     end
%     
%     %plot results for PM raw phase rms over time:
%     figure;
%     subplot(2,2,1)
%     plot(t_vec_peaks,PM_phase_raw_rms_over_time_vec,'b');
%     hold on;
%     plot(t_vec_peaks,mean(PM_phase_raw_rms_over_time_vec)*ones(length(t_vec_peaks),1),'r');
%     legend({'PM raw phase rms over time','mean raw phase rms over whole recording'},'location','best','fontsize',10);
%     title({'PM raw phase rms over time',strcat('Frequencies Filtered: ',num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),'[Hz]')});
%     xlabel('Time[Sec]');
%     subplot(2,2,2) 
%     plot(f_vec_peaks,abs(PM_phase_raw_rms_over_time_spectrum));
%     title('spectrum of PM raw phase rms over time');
%     xlabel('Frequency[Hz]');
%     subplot(2,2,3)
%     plot(t_vec_peaks_mirrored(3:end-2),auto_correlation_PM_phase_raw);
%     title('PM raw phase rms over time auto correlation');
%     subplot(2,2,4)
%     hist(PM_phase_raw_rms_over_time_vec,100);
%     title('histogram of PM raw phase rms');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' PM raw phase rms over time properties.jpg'),'jpg'); 
%     end
    
%     %plot results for PM filtered phase rms over time:
%     figure;
%     subplot(2,2,1)
%     plot(t_vec_peaks,PM_phase_filtered_rms_over_time_vec,'b');
%     hold on;
%     plot(t_vec_peaks,mean(PM_phase_filtered_rms_over_time_vec)*ones(length(t_vec_peaks),1),'r');
%     legend({'PM filtered phase rms over time','mean filtered phase rms over whole recording'},'location','best','fontsize',10);
%     title({'PM filtered phase rms over time',strcat('Frequencies Filtered: ',num2str(signal_start_frequency),'-',num2str(signal_stop_frequency),'[Hz]')});
%     xlabel('Time[Sec]');
%     subplot(2,2,2) 
%     plot(f_vec_peaks,abs(PM_phase_filtered_rms_over_time_spectrum));
%     title('spectrum of PM filtered phase rms over time');
%     xlabel('Frequency[Hz]');
%     subplot(2,2,3)
%     plot(t_vec_peaks_mirrored(3:end-2),auto_correlation_PM_phase_filtered);
%     title('PM filtered phase rms over time auto correlation');
%     subplot(2,2,4)
%     hist(PM_phase_filtered_rms_over_time_vec,100);
%     title('histogram of PM filtered phase rms');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' PM filtered phase rms over time properties.jpg'),'jpg'); 
%     end
    

%     %plot average logarithmic spectrums:
%     pause(1);
%     plot_BW=2000;
%     figure;
%     subplot(2,2,1)
%     plot(f_vec_average_spectrum,10*log10(average_spectrum_raw_signal(ceil(length(average_spectrum_raw_signal)/2)+(1-mod(length(average_spectrum_raw_signal),2)):end)));
%     xlim([Fc-plot_BW,Fc+plot_BW]);
%     title('mean raw signal spectrum [dB]');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD raw signal[dB]');
%     subplot(2,2,2)
%     plot(f_vec_average_spectrum,10*log10(average_spectrum_amplitude(ceil(length(average_spectrum_amplitude)/2)+(1-mod(length(average_spectrum_amplitude),2)):end)));
%     xlim([0,plot_BW]);
%     title('mean amplitude (from analytic absolute) signal spectrum [dB]');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD analytic amplitude[dB]');
%     subplot(2,2,3)
%     plot(f_vec_phase,10*log10(average_spectrum_phase_FM(ceil(length(average_spectrum_phase_FM)/2)+(1-mod(length(average_spectrum_phase_FM),2)):end)));
%     xlim([0,plot_BW]);
%     title('mean FM phase (from analytic angle) signal spectrum [dB]');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD FM phase[dB]');
%     subplot(2,2,4)
%     plot(f_vec_phase,10*log10(average_spectrum_phase_PM(ceil(length(average_spectrum_phase_FM)/2)+(1-mod(length(average_spectrum_phase_FM),2)):end)));
%     xlim([0,plot_BW]);
%     title('mean PM phase (from analytic angle) signal spectrum [dB]');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD PM phase[dB]');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' average logarithmic spectrums.jpg'),'jpg'); 
%     end
% 
%     %plot weighted average logarithmic spectrums:
%     figure;
%     subplot(2,2,1)
%     plot(f_vec_average_spectrum,10*log10(average_spectrum_raw_signal_weighted(ceil(length(average_spectrum_raw_signal_weighted)/2)+(1-mod(length(average_spectrum_raw_signal_weighted),2)):end)));
%     title('weighted (according to peak energy) mean raw signal spectrum');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD raw signal[dB]');
%     xlim([Fc-plot_BW,Fc+plot_BW]);
%     subplot(2,2,2)
%     plot(f_vec_average_spectrum,10*log10(average_spectrum_amplitude_weighted(ceil(length(average_spectrum_amplitude_weighted)/2)+(1-mod(length(average_spectrum_amplitude_weighted),2)):end)));
%     title('weighted (according to peak energy) mean amplitude (from analytic absolute) signal spectrum');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD analytic amplitude[dB]');
%     xlim([0,plot_BW]);
%     subplot(2,2,3)
%     plot(f_vec_phase,10*log10(average_spectrum_phase_FM_weighted(ceil(length(average_spectrum_phase_FM_weighted)/2)+(1-mod(length(average_spectrum_phase_FM_weighted),2)):end)));
%     title('weighted (according to peak energy) mean FM phase (from analytic angle) signal spectrum');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD FM phase[dB]');
%     xlim([0,plot_BW]);
%     subplot(2,2,4)
%     plot(f_vec_phase,10*log10(average_spectrum_phase_PM_weighted(ceil(length(average_spectrum_phase_FM_weighted)/2)+(1-mod(length(average_spectrum_phase_FM_weighted),2)):end)));
%     title('weighted (according to peak energy) mean PM phase (from analytic angle) signal spectrum');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD PM phase[dB]');
%     xlim([0,plot_BW]);
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' weighted logarithmic spectrums.jpg'),'jpg'); 
%     end

    
    %plot original linear spectrums with fitted gaussians:
    figure;
    subplot(2,2,1)
    scatter(zoomed_in_f_vec_averaged,zoomed_in_PSD_average,'b');
    hold on;
    plot(zoomed_in_f_vec_averaged,feval(fitted_parameters_average_spectrum,zoomed_in_f_vec_averaged),'r');
    xlabel('Frequency[Hz]');
    ylabel('Average PSD [linear]');
    legend({'original data','fitted gaussian'},'location','best','fontsize',10);
    title({strcat('BW = ',num2str(BW)),strcat('average spectrum fitted width: sigma = ',num2str(fitted_parameters_average_raw_spectrum_sigma),'[Hz]'),strcat('CNR (using energy integrals division)[dB] = ',num2str(CNR_averaged)),strcat('Carrier Above noise floor[dB] = ',num2str(CNR_above_noise_floor_with_RBW1Hz_averaged))...
        strcat('peak PSD = ',num2str(carrier_max_averaged,'%10.5e')),strcat('averaged noise floor (',num2str(effective_lobe_BW_one_sided),'-',num2str(BW),') = ',num2str(mean_noise_floor_averaged,'%10.2e'))});
    subplot(2,2,2)
    scatter(zoomed_in_f_vec_averaged_weighted,zoomed_in_PSD_averaged_weighted,'b');
    hold on;
    plot(zoomed_in_f_vec_averaged_weighted,feval(fitted_parameters_average_spectrum_weighted,zoomed_in_f_vec_averaged_weighted),'r');
    legend({'original data','fitted gaussian'},'location','best','fontsize',10);
    title({strcat('BW = ',num2str(BW)),strcat('weighted spectrum fitted width: sigma = ',num2str(fitted_parameters_average_raw_spectrum_weighted_sigma),'[Hz]'),strcat('CNR (using energy integrals division)[dB] = ',num2str(CNR_averaged_weighted)),strcat('Carrier Above noise floor[dB] = ',num2str(CNR_above_noise_floor_with_RBW1Hz_averaged_weighted))...
        strcat('peak PSD = ',num2str(carrier_max_averaged_weighted,'%10.5e')),strcat('averaged noise floor (',num2str(effective_lobe_BW_one_sided),'-',num2str(BW),') = ',num2str(mean_noise_floor_averaged_weighted,'%10.2e'))});
    xlabel('Frequency[Hz]');
    ylabel('Weighted PSD [linear]');
    %plot graph of CNR above noise for different frequencies:
    subplot(2,2,3)
    plot(f_vec_average_spectrum,CNR_above_noise_floor_with_RBW1Hz_averaged_vec(ceil(length(average_spectrum_raw_signal)/2)+(1-mod(length(average_spectrum_raw_signal),2)):end));
    ylim([0,max(max(CNR_above_noise_with_RBW1Hz_averaged_weighted_vec),max(CNR_above_noise_floor_with_RBW1Hz_averaged_vec))]);
    title('CNR above noise floor for RBW=1[Hz] vs. frequency');
    xlabel('Frequency[Hz]');
    ylabel('distance from carrier peak [dB]');
    subplot(2,2,4)
    plot(f_vec_average_spectrum,CNR_above_noise_with_RBW1Hz_averaged_weighted_vec(ceil(length(average_spectrum_raw_signal)/2)+(1-mod(length(average_spectrum_raw_signal),2)):end));
    ylim([0,max(max(CNR_above_noise_with_RBW1Hz_averaged_weighted_vec),max(CNR_above_noise_floor_with_RBW1Hz_averaged_vec))]);
    title('CNR above noise floor for RBW=1[Hz] vs. frequency');
    xlabel('Frequency[Hz]');
    ylabel('distance from carrier peak [dB]');
    super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
    if flag_save_plots==1
       saveas(gcf,strcat(fullfile(directory,raw_file_name),' raw spectrums with fitted gaussians.jpg'),'jpg'); 
    end
    close('all');
    
%     %plot PSD energy percentage as a function of offset from carrier:
%     energy_percentage_vec_from_carrier_average = energy_vec_from_carrier_average/max(energy_vec_from_carrier_average)*100;
%     energy_percentage_vec_from_carrier_average_weighted = energy_vec_from_carrier_average_weighted/max(energy_vec_from_carrier_average_weighted)*100;
%     
%     f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(energy_percentage_vec_from_carrier_average),0,BW/3]);
%     ft = fittype('A*(1-1/(1+abs(x-b)/sigma))','options',f0);
%     [fitted_parameters_percentage_average,bla1] = fit(frequency_offset_from_carrier_vec_average(:),energy_percentage_vec_from_carrier_average(:),ft);
%     energy_percentage_vs_offset_from_carrier_lorentzian_fit_sigma1 = fitted_parameters_percentage_average.sigma;
%     
%     f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(energy_percentage_vec_from_carrier_average_weighted),0,BW/3]);
%     ft = fittype('A*(1-1/(1+abs(x-b)/sigma))','options',f0);
%     [fitted_parameters_percentage_weighted,bla1] = fit(frequency_offset_from_carrier_vec_average_weighted(:),energy_percentage_vec_from_carrier_average_weighted(:),ft);
%     energy_percentage_vs_offset_from_carrier_lorentzian_fit_sigma2 = fitted_parameters_percentage_weighted.sigma;
    
%     figure;
%     subplot(2,1,1)
%     scatter(frequency_offset_from_carrier_vec_average,energy_percentage_vec_from_carrier_average);
%     hold on;
%     plot(frequency_offset_from_carrier_vec_average,feval(fitted_parameters_percentage_average,frequency_offset_from_carrier_vec_average),'r');
%     legend({'data','fit'},'location','best','fontsize',10);
%     title({'PSD energy percentage as a function of offset from carrier average',texlabel(fitted_parameters_percentage_average),strcat('sigma=',num2str(energy_percentage_vs_offset_from_carrier_lorentzian_fit_sigma1))});
%     xlabel('distance from carrier [Hz]');
%     ylabel('percentage of total energy covered');
%     subplot(2,1,2)
%     scatter(frequency_offset_from_carrier_vec_average_weighted,energy_percentage_vec_from_carrier_average_weighted);
%     hold on;
%     plot(frequency_offset_from_carrier_vec_average_weighted,feval(fitted_parameters_percentage_average,frequency_offset_from_carrier_vec_average_weighted),'r');
%     legend({'data','fit'},'location','best','fontsize',10);
%     title({'PSD energy percentage as a function of offset from carrier average weighted',texlabel(fitted_parameters_percentage_weighted),strcat('sigma=',num2str(energy_percentage_vs_offset_from_carrier_lorentzian_fit_sigma2))});
%     xlabel('distance from carrier [Hz]');
%     ylabel('percentage of total energy covered');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' energy percentage as a function of offset from carrier.jpg'),'jpg'); 
%     end
%     
%     
%     %plot spectrum of frame with max fft value:
%     [relevant_PSD_max,relevant_f_vec_max] = get_fft_between_certain_frequencies(max_spectrum,Fs,3,2,noise_floor_left_start_averaged,noise_floor_right_stop_averaged);
%     figure;
%     subplot(2,1,1)
%     scatter(relevant_f_vec_max,relevant_PSD_max,'b');
%     hold on;
%     plot(zoomed_in_f_vec_averaged,feval(fitted_parameters_average_spectrum,zoomed_in_f_vec_averaged),'r');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD [linear]');
%     legend({'original data','fitted gaussian'},'location','best','fontsize',10);
%     title({strcat('BW = ',num2str(BW)),strcat('max CNR (using energy integrals division)[dB] = ',num2str(CNR_energy_division_max)),strcat('Carrier Above noise floor[dB] = ',num2str(CNR_above_noise_floor_with_RBW1Hz_max))...
%         ,strcat('max spectrum fitted width: sigma = ',num2str(fitted_parameters_max_spectrum.sigma),'[Hz]'),strcat('peak PSD = ',num2str(max_value^2,'%10.5e')),strcat('averaged noise floor (',num2str(effective_lobe_BW_one_sided),'-',num2str(BW),') = ',num2str(mean_noise_floor_PSD_max,'%10.2e'))});
%     subplot(2,1,2)
%     plot(relevant_f_vec_max,10*log10(relevant_PSD_max),'g');
%     xlabel('Frequency[Hz]');
%     ylabel('PSD [dB]');
%     title('10*log10(PSD)');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' max fft CNR and PSD.jpg'),'jpg'); 
%     end
%     
%     
%     %plot linear CNR's over time and histograms:
%     figure;
%     subplot(2,2,1) 
%     plot(t_vec_peaks,CNR_energy_division_over_time_linear_vec,'r',t_vec_peaks,mean(CNR_energy_division_over_time_linear_vec)*ones(length(t_vec_peaks),1),'r')
%     legend({'CNR over time','mean CNR'},'location','best','fontsize',10);
%     xlabel('Time[sec]');
%     title({'linear CNR (energy divisions) over time',strcat('mean CNR = ', num2str(mean_linear_CNR_energy_division)),strcat('std CNR around mean = ',num2str(std_linear_CNR_energy_division))});
%     subplot(2,2,2)
%     plot(t_vec_peaks,10.^(CNR_above_noise_floor_over_time_log_vec/10));
%     xlabel('Time[sec]');
%     title('linear CNR (above noise floor) over time');
%     subplot(2,2,3)
%     hist(10.^(CNR_energy_division_over_time_log_vec/10),100);
%     title('linear CNR (energy division) histogram');
%     subplot(2,2,4)
%     hist(10.^(CNR_above_noise_floor_over_time_log_vec/10),100);
%     title('linear CNR (above noise floor) histogram');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' linear CNR over time and histogram.jpg'),'jpg'); 
%     end
%     
%     %plot log CNR's over time and histograms:
%     figure;
%     subplot(2,2,1)
%     plot(t_vec_peaks,CNR_energy_division_over_time_log_vec)
%     title({'log CNR (energy divisions) over time',strcat('mean CNR = ', num2str(mean_log_CNR_energy_division)),strcat('std CNR around mean = ',num2str(std_log_CNR_energy_division))});
%     subplot(2,2,2)
%     plot(t_vec_peaks,CNR_above_noise_floor_over_time_log_vec);
%     title('log CNR (above noise floor) over time');
%     subplot(2,2,3)
%     hist(CNR_energy_division_over_time_log_vec,100);
%     title('log CNR (energy division) histogram');
%     subplot(2,2,4)
%     hist(CNR_above_noise_floor_over_time_log_vec,100);
%     title('log CNR (above noise floor) histogram');
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' log CNR over time and histogram.jpg'),'jpg'); 
%     end
% 
%     %plot raw signal and recovered analytic signal amplitude over time:
%     if flag_write_binary_files_of_amplitude_and_phase_and_later==1
%         number_of_seconds_to_read = 2;
%         number_of_elements_to_read = round(Fs*number_of_seconds_to_read);
%         fid_amplitude = fopen(full_file_name_amplitude,'r');
%         fid_phase = fopen(full_file_name_phase,'r');
%         fid = fopen(full_file_name,'r');
%         temp=fread(fid,2,'double');
%         temp=fread(fid_amplitude,2,'double');
%         temp=fread(fid_phase,2,'double');
%         raw_vec = fread(fid,number_of_elements_to_read,'double');
%         amplitude_vec = fread(fid_amplitude,number_of_elements_to_read,'double');
%         phase_vec = fread(fid_phase,number_of_elements_to_read,'double');
%         fclose(fid);
%         fclose(fid_amplitude);
%         fclose(fid_phase);
%         %raw signal and analytic signal amplitude
%         figure(8)
%         subplot(2,1,1)
%         plot(mylinspace(0,1/Fs,length(raw_vec)),raw_vec);
%         title('raw signal');
%         subplot(2,1,2)
%         plot(mylinspace(0,1/Fs,length(amplitude_vec)),amplitude_vec);
%         title('analytic signal amplitude vec');
%         if flag_save_plots==1
%            saveas(gcf,strcat(fullfile(directory,raw_file_name),' plot8.jpg'),'jpg'); 
%         end
%     end
% 
%     %plot peak fourier frequency, peak to peak, rms, and total PSD energy over time:
%     figure;
%     subplot(2,2,1)
%     plot(t_vec_peaks,peak_fourier_value_over_time_vec);
%     title('peak fft value over time');
%     subplot(2,2,2)
%     plot(t_vec_peaks,PSD_total_energy_over_time_vec);
%     title('total PSD energy over time');
%     subplot(2,2,3)
%     plot(t_vec_peaks,raw_p2p_over_time_vec);
%     title({'peak to peak over time',strcat('average peak to peak = ',num2str(mean(raw_p2p_over_time_vec)))});
%     subplot(2,2,4)
%     plot(t_vec_peaks,raw_rms_over_time_vec);
%     title({'RMS over time',strcat('average rms of signal = ',num2str(mean(raw_rms_over_time_vec)))});
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' fourier peak, p2p, rms, PSD over time.jpg'),'jpg'); 
%     end
% 
%     %plot phase signal:
%     if flag_write_binary_files_of_amplitude_and_phase_and_later==1
%         PM_phase_vec = phase_vec(samples_per_frame*3:end);
%         t_vec_phase = mylinspace(0,1/Fs,length(phase_vec));
%         t_vec_phase = t_vec_phase(samples_per_frame*3:end);
%         FM_phase_vec=PM_phase_vec(2:end)-PM_phase_vec(1:end-1);
%         figure(10)
%         subplot(3,1,1)
%         plot(t_vec_phase,PM_phase_vec);
%         title('analytic signal phase (PM) vec');
%         subplot(3,1,2)
%         plot(t_vec_phase(1:length(t_vec_phase)-1),FM_phase_vec);
%         title('analytic signal phase difference (FM) vec');
%         legend(strcat('Standard deviation of signal phase (FM) over time = ',num2str(std(FM_phase_vec))));
%         subplot(3,1,3)
%         plot(t_vec_peaks,peak_fourier_frequency_vec);
%         title('peak frequency over time (to see how stable it the carrier peak frequency)');
%         legend(strcat('standard deviation of peak frequency over time = ',num2str(std(peak_fourier_frequency_vec(1:round(length(peak_fourier_frequency_vec)/2)))),'[Hz]'));
%         super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%         if flag_save_plots==1
%            saveas(gcf,strcat(fullfile(directory,raw_file_name),' plot10.jpg'),'jpg'); 
%         end
%     end
%     
%     %plot PM and FM phase rms over time:
%     if flag_analyze_phase_rms_over_time==1
%         figure;
%         subplot(2,1,1)
%         plot(t_vec_peaks,PM_phase_filtered_rms_over_time_vec);
%         title({'PM phase rms over time',strcat('PM phase total rms = ',num2str(sqrt(sum(PM_phase_filtered_rms_over_time_vec.^2))/sqrt(length(PM_phase_filtered_rms_over_time_vec))))});
%         xlabel('Time[sec]');
%         ylabel('phase RMS');
%         subplot(2,1,2)
%         plot(t_vec_peaks,FM_phase_filtered_rms_over_time_vec);
%         title({'FM phase rms over time',strcat('FM phase total rms = ',num2str(sqrt(sum(FM_phase_filtered_rms_over_time_vec.^2))/sqrt(length(FM_phase_filtered_rms_over_time_vec))))});
%         xlabel('Time[sec]');
%         ylabel('phase RMS');
%         super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%         if flag_save_plots==1
%            saveas(gcf,strcat(fullfile(directory,raw_file_name),' phase RMS over time FM and PM.jpg'),'jpg'); 
%         end 
%     end
%     
%     %plot spectrum energies over different bandwidths:
%     list = hsv(size(fft_energy_in_frequency_bins,1));
%     figure;
%     subplot(ceil(size(fft_energy_in_frequency_bins,1)/2),2,1)
%     plot(t_vec_peaks,peak_fourier_value_over_time_vec,'color',list(1,:));
%     legend('carrier peak over time');
%     for k=2:length(list)
%         subplot(ceil(size(fft_energy_in_frequency_bins,1)/2),2,k)
%         plot(t_vec_peaks,fft_energy_in_frequency_bins(k,:),'color',list(k,:));
%         hold on; 
%         if k<length(list)
%             legend(strcat('BW=',num2str(BW_bins_around_mean_carrier_frequency(k)),'[Hz]'),'fontsize',8);
%         else
%             legend('total PSD energy (entier BW)','fontsize',8);
%         end
%         title('PSD energy over time for different BW');
%     end
%     super_title(strcat('Frame Time = ',num2str(time_for_FFT_frame)),20);
%     if flag_save_plots==1
%        saveas(gcf,strcat(fullfile(directory,raw_file_name),' PSD energy over different bandwidths.jpg'),'jpg'); 
%     end
%     close('all');  
    
    






















% 
%     %Append current variables to Excel:
%     percentage_of_samples = mean(number_of_spikes_vec)/time_for_FFT_frame/Fs;
%     xlsx_file_name = 'SOL turbulence and stability experiment new2.xlsx';
%    
%     xlsx_headers = {...
%         'file name',...
%         'comments',...
%         'hour',...
%         'raw RMS',...
%         'FM phase raw rms',...
%         'PM phase raw rms',...
%         'FM phase raw rms after mask',... 
%         'PM phase raw rms after mask',...
%         'FM phase filtered rms',...
%         'PM phase filtered rms',...
%         'FM phase filtered rms after mask',...
%         'PM phase filtered rms after mask',...
%         'average carrier linear lorentzian fit sigma',...
%         'average FM phase linear lorentzian fit sigma',...
%         'average PM phase linear lorentzian fit sigma',...
%         'average analytical amplitude linear lorentzian fit sigma',...
%         'energy percentage vs offset from carrier sigma',...
%         'max CNR energy division',...
%         'CNR energy division using mean spectrum',...
%         'CNR above noise using mean spectrum',...
%         'mean linear CNR energy division over time',...
%         'mean log CNR energy division over time',...
%         'std of linear CNR energy division over time',...
%         'average noise floor',...
%         'percentage of samples'};
%     
%     
%     xlsx_data = {...
%         raw_file_name,... %file name
%         [],... %comments to be filled by me
%         [],... %hour to be filled by me
%         rms(raw_rms_over_time_vec),... %raw rms
%         rms(FM_phase_raw_rms_over_time_vec),... %FM phase raw rms
%         rms(PM_phase_raw_rms_over_time_vec),... %PM phase raw rms
%         rms(FM_phase_raw_rms_over_time_after_mask_vec),... %FM phase raw rms after mask
%         rms(PM_phase_raw_rms_over_time_after_mask_vec),... %PM phase raw rms after mask
%         rms(FM_phase_filtered_rms_over_time_vec),... %FM phase filtered rms
%         rms(PM_phase_filtered_rms_over_time_vec),... %PM phase filtered rms
%         rms(FM_phase_filtered_rms_over_time_after_mask_vec),... %FM phase filtered after mask rms
%         rms(PM_phase_filtered_rms_over_time_after_mask_vec),... %PM phase filtered after mask rms
%         average_carrier_linear_lorentzian_fit_sigma,... %average carrier linear fit sigma
%         average_FM_phase_linear_lorentzian_fit_sigma,... %average FM phase linear fit sigma
%         average_PM_phase_linear_lorentzian_fit_sigma,... %average PM phase linear fit sigma
%         average_analytical_amplitude_linear_lorentzian_fit_sigma,... %average analytical amplitude linear fit sigma
%         energy_percentage_vs_offset_from_carrier_lorentzian_fit_sigma1,... %energy percentage from carrier fit sigma
%         CNR_energy_division_max,... %CNR energy division max
%         CNR_averaged,... %CNR energy division average from mean spectrum
%         CNR_above_noise_floor_with_RBW1Hz_averaged,... %CNR above noise from mean spectrum
%         mean_linear_CNR_energy_division,... %mean linear CNR from CNR over time vec
%         mean_log_CNR_energy_division,... %mean log CNR from CNR over time vec
%         std_linear_CNR_energy_division,... %std of linear CNR from CNR over time vec
%         mean_noise_floor_averaged,... %mean noise floor of PSD
%         percentage_of_samples}; %average number of spikes per second
%        
%     xlsx_sheet=1;
%     
%     xlsappend(xlsx_file_name,xlsx_data,xlsx_sheet);
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

%RESULTS GRAPHS:
if flag_try_compressed_and_clipped_carrier==1
        %calculate spectrum of fourier peaks:
        t_vec_peaks = 0:samples_per_frame/Fs:samples_per_frame/Fs*(length(peak_fourier_value_compressed_and_clipped_vec)-1);
        t_vec_peaks_mirrored = mirror_array_append(t_vec_peaks,1);
        Fs_peaks = 1/(t_vec_peaks(2)-t_vec_peaks(1));
        [f_vec_average_spectrum] = fft_get_frequency_vec(samples_per_frame,Fs,1);
        [fft_peaks_compressed_and_clipped_spectrum,f_vec_peaks] = calculate_simple_fft(peak_fourier_value_compressed_and_clipped_vec,Fs_peaks,2,0);
        matlab_auto_correlation_peaks_compressed_and_clipped = autocorr(peak_fourier_value_compressed_and_clipped_vec,length(peak_fourier_value_compressed_and_clipped_vec)-1);
        matlab_auto_correlation_peaks_compressed_and_clipped = mirror_array_append(matlab_auto_correlation_peaks_compressed_and_clipped,0);
        %normalize averaged and weighted spectrums by number of frames to compare it to others:
        average_spectrum_raw_signal_compressed_and_clipped = average_spectrum_raw_signal_compressed_and_clipped/counter;
        average_spectrum_amplitude_compressed_and_clipped = average_spectrum_amplitude_compressed_and_clipped/counter;
        average_spectrum_phase_compressed_and_clipped = average_spectrum_phase_compressed_and_clipped/counter;
        average_spectrum_raw_signal_weighted_compressed_and_clipped = average_spectrum_raw_signal_weighted_compressed_and_clipped/counter;
        average_spectrum_amplitude_weighted_compressed_and_clipped = average_spectrum_amplitude_weighted_compressed_and_clipped/counter;
        average_spectrum_phase_weighted_compressed_and_clipped = average_spectrum_phase_weighted_compressed_and_clipped/counter;

        %plot results for fourier peak over time:
        figure(1)
        subplot(2,2,1)
        plot(t_vec_peaks,peak_fourier_value_compressed_and_clipped_vec);
        title('peak fourier component over time');
        subplot(2,2,2)
        plot(f_vec_peaks,abs(fft_peaks_compressed_and_clipped_spectrum));
        title('fft of fourier peaks over time');
        subplot(2,2,3)
        plot(t_vec_peaks_mirrored,matlab_auto_correlation_peaks_compressed_and_clipped);
        title('matlabs autocorrelation');
        subplot(2,2,4)
        hist(peak_fourier_value_compressed_and_clipped_vec,100);
        title('histogram of carrier fft peak');
        if flag_save_plots==1
           saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot1.jpg'),'jpg'); 
        end

        %plot average spectrums:
        pause(1);
        figure(2)
        subplot(3,1,1)
        plot(f_vec_average_spectrum,10*log10(average_spectrum_raw_signal_compressed_and_clipped(ceil(length(average_spectrum_raw_signal_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_raw_signal_compressed_and_clipped),2)):end)));
        title('mean raw signal spectrum');
        subplot(3,1,2)
        plot(f_vec_average_spectrum,10*log10(average_spectrum_amplitude_compressed_and_clipped(ceil(length(average_spectrum_amplitude_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_amplitude_compressed_and_clipped),2)):end)));
        title('mean amplitude (from analytic absolute) signal spectrum');
        subplot(3,1,3)
        plot(f_vec_average_spectrum,10*log10(average_spectrum_phase_compressed_and_clipped(ceil(length(average_spectrum_phase_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_phase_compressed_and_clipped),2)):end)));
        title('mean phase (from analytic angle) signal spectrum');
        if flag_save_plots==1
           saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot2.jpg'),'jpg'); 
        end

        %plot weighted average spectrums:
        figure(3)
        subplot(3,1,1)
        plot(f_vec_average_spectrum,10*log10(average_spectrum_raw_signal_weighted_compressed_and_clipped(ceil(length(average_spectrum_raw_signal_weighted_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_raw_signal_weighted_compressed_and_clipped),2)):end)));
        title('weighted (according to peak energy) mean raw signal spectrum');
        subplot(3,1,2)
        plot(f_vec_average_spectrum,10*log10(average_spectrum_amplitude_weighted_compressed_and_clipped(ceil(length(average_spectrum_amplitude_weighted_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_amplitude_weighted_compressed_and_clipped),2)):end)));
        title('weighted (according to peak energy) mean amplitude (from analytic absolute) signal spectrum');
        subplot(3,1,3)
        plot(f_vec_average_spectrum,10*log10(average_spectrum_phase_weighted_compressed_and_clipped(ceil(length(average_spectrum_phase_weighted_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_phase_weighted_compressed_and_clipped),2)):end)));
        title('weighted (according to peak energy) mean phase (from analytic angle) signal spectrum');
        if flag_save_plots==1
           saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot3.jpg'),'jpg'); 
        end

        %fit a gaussian to the carrier lobe and calculate effective CNR:
        %fit gaussian to averaged and weighted lobes:
        f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(average_spectrum_raw_signal_compressed_and_clipped),average_max_frequency,200]);
        ft = fittype('A*exp(-(x-b)^2/(2*sigma^2))','options',f0);
        [fitted_parameters_average_spectrum_compressed,bla1] = fit(f_vec_average_spectrum(:),average_spectrum_raw_signal_compressed_and_clipped(ceil(length(average_spectrum_raw_signal_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_raw_signal_compressed_and_clipped),2)):end),ft);
        f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(average_spectrum_raw_signal_weighted_compressed_and_clipped),average_max_frequency,200]);
        ft = fittype('A*exp(-(x-b)^2/(2*sigma^2))','options',f0);
        [fitted_parameters_average_spectrum_weighted_compressed,bla2] = fit(f_vec_average_spectrum(:),average_spectrum_raw_signal_weighted_compressed_and_clipped(ceil(length(average_spectrum_raw_signal_weighted_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_raw_signal_weighted_compressed_and_clipped),2)):end),ft);
        f0 = fitoptions('Method','NonlinearLeastSquares','StartPoint',[max(average_spectrum_raw_signal_weighted_compressed_and_clipped),average_max_frequency,200]);
        ft = fittype('A*exp(-(x-b)^2/(2*sigma^2))','options',f0);
        [fitted_parameters_max_spectrum_compressed,bla3] = fit(f_vec_average_spectrum(:),max_spectrum_compressed_and_clipped(ceil(length(max_spectrum_compressed_and_clipped)/2)+(1-mod(length(max_spectrum_compressed_and_clipped),2)):end),ft);
        %calculate average CNRs of averaged spectrum:
        BW = 5000;
        [CNR_averaged_compressed,CNR_above_noise_with_RBW1Hz_averaged_compressed,CNR_above_noise_with_RBW1Hz_averaged_vec_compressed,mean_noise_floor_averaged_compressed,carrier_max_averaged_compressed] = ...
            calculate_CNR(average_spectrum_raw_signal_compressed_and_clipped,3,BW,effective_lobe_BW_one_sided,Fs,average_max_frequency);
        [carrier_lobe_start_averaged_compressed,carrier_lobe_stop_averaged_compressed,noise_floor_left_start_averaged_compressed,...
        noise_floor_left_stop_averaged_compressed,noise_floor_right_start_averaged_compressed,noise_floor_right_stop_averaged_compressed] = ...
        fft_get_carrier_lobe_frequencies(average_max_frequency,effective_lobe_BW_one_sided,BW);  
        %calculate average CNRs of averaged and weighted spectrum:
        BW = 5000;
        [CNR_averaged_weighted_compressed,CNR_above_noise_with_RBW1Hz_averaged_weighted_compressed,CNR_above_noise_with_RBW1Hz_averaged_weighted_vec_compressed,mean_noise_floor_averaged_weighted_compressed,carrier_max_averaged_weighted_compressed] = ...
            calculate_CNR(average_spectrum_raw_signal_weighted_compressed_and_clipped,3,BW,effective_lobe_BW_one_sided,Fs,average_max_frequency);
        [carrier_lobe_start_averaged_weighted_compressed,carrier_lobe_stop_averaged_weighted_compressed,noise_floor_left_start_averaged_weighted_compressed,...
        noise_floor_left_stop_averaged_weighted_compressed,noise_floor_right_start_averaged_weighted_compressed,noise_floor_right_stop_averaged_weighted_compressed] = ...
        fft_get_carrier_lobe_frequencies(average_max_frequency,effective_lobe_BW_one_sided,BW);   
        %get relevant PSDs and f_vecs for plotting:
        [relevant_PSD_averaged_compressed,relevant_f_vec_averaged_compressed] = get_fft_between_certain_frequencies(average_spectrum_raw_signal_compressed_and_clipped,Fs,3,2,noise_floor_left_start_averaged,noise_floor_right_stop_averaged);
        [relevant_PSD_averaged_weighted_compressed,relevant_f_vec_averaged_weighted_compressed] = get_fft_between_certain_frequencies(average_spectrum_raw_signal_weighted_compressed_and_clipped,Fs,3,2,noise_floor_left_start_averaged,noise_floor_right_stop_averaged);
        %get PSD energy as a function of offset from carrier:
        [energy_vec_from_carrier_average_compressed,frequency_offset_from_carrier_vec_average_compressed] = fft_get_PSD_energy_vs_distance_from_carrier(average_spectrum_raw_signal_compressed_and_clipped,Fs,average_max_frequency,BW,3,3,2);
        [energy_vec_from_carrier_average_weighted_compressed,frequency_offset_from_carrier_vec_average_weighted_compressed] = fft_get_PSD_energy_vs_distance_from_carrier(average_spectrum_raw_signal_weighted_compressed_and_clipped,Fs,average_max_frequency,BW,3,3,2);

        %plot original linear spectrums with fitted gaussians:
        figure(4)
        subplot(2,2,1)
        scatter(zoomed_in_f_vec_averaged,relevant_PSD_averaged_compressed,'b');
        hold on;
        plot(zoomed_in_f_vec_averaged,feval(fitted_parameters_average_spectrum_compressed,zoomed_in_f_vec_averaged),'r');
        legend({'original data','fitted gaussian'});
        title({strcat('BW = ',num2str(BW)),strcat('weighted spectrum fitted width: sigma = ',num2str(fitted_parameters_average_spectrum_compressed.sigma),'[Hz]'),strcat('CNR (using energy integrals division)[dB] = ',num2str(CNR_averaged_compressed)),strcat('Carrier Above noise floor[dB] = ',num2str(CNR_above_noise_with_RBW1Hz_averaged_compressed))...
            strcat('peak PSD = ',num2str(carrier_max_averaged_compressed,'%10.5e')),strcat('averaged noise floor (',num2str(effective_lobe_BW_one_sided),'-',num2str(BW),') = ',num2str(mean_noise_floor_averaged_compressed,'%10.2e'))});
        subplot(2,2,2)
        scatter(zoomed_in_f_vec_averaged_weighted,relevant_PSD_averaged_weighted_compressed,'b');
        hold on;
        plot(zoomed_in_f_vec_averaged_weighted,feval(fitted_parameters_average_spectrum_weighted_compressed,zoomed_in_f_vec_averaged_weighted),'r');
        legend({'original data','fitted gaussian'});
        title({strcat('BW = ',num2str(BW)),strcat('weighted spectrum fitted width: sigma = ',num2str(fitted_parameters_average_spectrum_weighted_compressed.sigma),'[Hz]'),strcat('CNR (using energy integrals division)[dB] = ',num2str(CNR_averaged_weighted_compressed)),strcat('Carrier Above noise floor[dB] = ',num2str(CNR_above_noise_with_RBW1Hz_averaged_weighted_compressed))...
            strcat('peak PSD = ',num2str(carrier_max_averaged_weighted_compressed,'%10.5e')),strcat('averaged noise floor (',num2str(effective_lobe_BW_one_sided),'-',num2str(BW),') = ',num2str(mean_noise_floor_averaged_weighted_compressed,'%10.2e'))});
        %plot graph of CNR above noise for different frequencies:
        subplot(2,2,3)
        plot(f_vec_average_spectrum,CNR_above_noise_with_RBW1Hz_averaged_vec_compressed(ceil(length(average_spectrum_raw_signal_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_raw_signal_compressed_and_clipped),2)):end));
        title('CNR above noise floor for RBW=1[Hz] vs. frequency');
        subplot(2,2,4)
        plot(f_vec_average_spectrum,CNR_above_noise_with_RBW1Hz_averaged_weighted_vec_compressed(ceil(length(average_spectrum_raw_signal_compressed_and_clipped)/2)+(1-mod(length(average_spectrum_raw_signal_compressed_and_clipped),2)):end));
        title('CNR above noise floor for RBW=1[Hz] vs. frequency');
        super_title(strcat('Fc = ',num2str(average_max_frequency)),12);
        if flag_save_plots==1
           saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot4.jpg'),'jpg'); 
        end

        %plot PSD energy as a function of offset from carrier:
        figure(5)
        subplot(2,1,1)
        plot(frequency_offset_from_carrier_vec_average_compressed,energy_vec_from_carrier_average_compressed);
        title('PSD energy as a function of offset from carrier average');
        subplot(2,1,2)
        plot(frequency_offset_from_carrier_vec_average_weighted_compressed,energy_vec_from_carrier_average_weighted_compressed);
        title('PSD energy as a function of offset from carrier average weighted');
        if flag_save_plots==1
           saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot5.jpg'),'jpg'); 
        end

        %plot spectrum of frame with max fft value:
        [relevant_PSD_max_compressed_and_clipped,relevant_f_vec_max] = get_fft_between_certain_frequencies(max_spectrum_compressed_and_clipped,Fs,3,2,noise_floor_left_start_averaged,noise_floor_right_stop_averaged);
        figure(6)
        subplot(2,1,1)
        plot(relevant_f_vec_max,relevant_PSD_max_compressed_and_clipped,'b');
        title({strcat('BW = ',num2str(BW)),strcat('CNR (using energy integrals division)[dB] = ',num2str(CNR_max_compressed_and_clipped)),strcat('Carrier Above noise floor[dB] = ',num2str(CNR_above_noise_floor_with_RBW1Hz_max_compressed_and_clipped))...
            strcat('peak PSD = ',num2str(max_value_compressed_and_clipped^2,'%10.5e')),strcat('averaged noise floor (',num2str(effective_lobe_BW_one_sided),'-',num2str(BW),') = ',num2str(mean_noise_floor_max_compressed_and_clipped,'%10.2e'))});
        subplot(2,1,2)
        plot(relevant_f_vec_max,10*log10(relevant_PSD_max_compressed_and_clipped),'g');
        title('10*log10(PSD)');
        if flag_save_plots==1
           saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot6.jpg'),'jpg'); 
        end

        %plot peak fourier frequency, peak to peak, rms, and total PSD energy over time:
        figure(8)
        subplot(2,2,1)
        plot(t_vec_peaks,peak_fourier_value_compressed_and_clipped_vec);
        title('peak fft value over time');
        subplot(2,2,2)
        plot(t_vec_peaks,total_PSD_energy_compressed_and_clipped);
        title('total PSD energy over time');
        subplot(2,2,3)
        plot(t_vec_peaks,p2p_compressed_and_clipped_vec);
        title({'peak to peak over time',strcat('average peak to peak = ',num2str(mean(p2p_compressed_and_clipped_vec)))});
        subplot(2,2,4)
        plot(t_vec_peaks,rms_compressed_and_clipped_vec);
        title({'RMS over time',strcat('average rms of signal = ',num2str(mean(rms_compressed_and_clipped_vec)))});
        if flag_save_plots==1
           saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot8.jpg'),'jpg'); 
        end

        %plot PM and FM phase rms over time:
        if flag_analyze_phase_rms_over_time==1
            figure(10)
            subplot(2,1,1)
            plot(t_vec_peaks,PM_phase_rms_compressed_and_clipped_vec);
            title({'PM phase rms over time',strcat('PM phase total rms = ',num2str(sqrt(sum(PM_phase_rms_compressed_and_clipped_vec.^2))/sqrt(length(PM_phase_rms_compressed_and_clipped_vec))))});
            subplot(2,1,2)
            plot(t_vec_peaks,FM_phase_rms_compressed_and_clipped_vec);
            title({'FM phase rms over time',strcat('FM phase total rms = ',num2str(sqrt(sum(FM_phase_rms_compressed_and_clipped_vec.^2))/sqrt(length(FM_phase_rms_compressed_and_clipped_vec))))});
            if flag_save_plots==1
               saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot10.jpg'),'jpg'); 
            end 
        end

        %plot spectrum energies over different bandwidths:
        list = hsv(size(fft_energy_in_frequency_bins_compressed_and_clipped,1));
        figure(11)
        subplot(ceil(size(fft_energy_in_frequency_bins_compressed_and_clipped,1)/2),2,1)
        plot(t_vec_peaks,peak_fourier_value_compressed_and_clipped_vec,'color',list(1,:));
        ylim([min(peak_fourier_frequency_compressed_and_clipped_vec(1:end-10)),max(peak_fourier_frequency_compressed_and_clipped_vec(1:end-10))]);
        legend('carrier peak over time');
        for k=2:length(list)
            subplot(ceil(size(fft_energy_in_frequency_bins_compressed_and_clipped,1)/2),2,k)
            plot(t_vec_peaks,fft_energy_in_frequency_bins_compressed_and_clipped(k,:),'color',list(k,:));
            ylim([min(fft_energy_in_frequency_bins_compressed_and_clipped(k,1:end-10)),max(fft_energy_in_frequency_bins_compressed_and_clipped(k,1:end-10))]);
            hold on;
            if k<length(list)
                legend(strcat('BW=',num2str(BW_bins_around_mean_carrier_frequency(k)),'[Hz]'));
            else
                legend('total PSD energy (entier BW)');
            end
            title('PSD energy over time for different BW');
        end
        if flag_save_plots==1
           saveas(gcf,strcat(fullfile(directory,raw_file_name),'compressed and clipped plot11.jpg'),'jpg'); 
        end
        close('all'); 
end  
    
end  

toc
end





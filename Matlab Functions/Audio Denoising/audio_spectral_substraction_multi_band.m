function [] = audio_spectral_substraction_multi_band(input_file_name,output_file_name,number_of_frequency_bands,frequency_scale)

%Explanations:
% flag_use_VAD -> Use voice activity detector, choices: 1 -to use VAD and 0 -otherwise
% flag_do_pre_smoothing_and_averaging -> Do pre-processing (smoothing & averaging), choice: 1 -for pre-processing and 0 -otherwise, default=1
% frame_size_in_seconds -> Frame length in milli-seconds, default=20
% frame_overlap_percentage -> Window overlap in percent of frame size, default=50
% number_of_noise_frames_at_file_start -> Number of noise frames at beginning of file for noise spectrum estimate, default=6
% spectral_floor -> Spectral floor, default=0.002

%Input Variables:
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio170-3500[Hz]lsq smoother with deframing';
output_file_name = 'rodena_speaking_spectral_subtraction.wav';
number_of_frequency_bands = 6;
frequency_scale = 'linear';

%Flags:
flag_do_pre_smoothing_and_averaging=1;
flag_use_VAD=1;

%Audio parameters:
frame_size_in_mili_seconds=20; 
frame_overlap_percentage=50; 
number_of_noise_frames_at_file_start=6; 
spectral_floor=0.002; 

%Read file:
[input_signal,Fs] = wavread(input_file_name);
input_signal = input_signal(:,1); %single channel
input_signal = add_noise_of_certain_SNR(input_signal,5,1,0);

%Further Audio Parameters:
samples_per_frame = floor(frame_size_in_mili_seconds*Fs/1000);   
number_of_overlap_samples_per_frame = floor(samples_per_frame*frame_overlap_percentage/100);        % Number of overlap samples
number_of_common_samples_between_frames = samples_per_frame-number_of_overlap_samples_per_frame;              % Number of common samples between adjacent frames
fft_length = 2.^nextpow2(samples_per_frame);

%Choose Frequency Vec:
switch frequency_scale
case {'linear','LINEAR'}
    [bin_start,bin_center,bin_stop,bin_size] = fft_get_linear_frequency_vecs(number_of_frequency_bands,fft_length);
case {'log','LOG'}
    [bin_start,bin_center,bin_stop,bin_size] = fft_get_log_frequency_vecs(number_of_frequency_bands,Fs,fft_length);
case {'mel','MEL'}
    [bin_start,bin_center,bin_stop,bin_size] = fft_get_mel_frequency_vecs(number_of_frequency_bands,0,Fs/2,fft_length,Fs);
otherwise
    fprintf('Error in selecting frequency spacing, type "help mbss" for help.\n');
    return; 
end
     
%Calculate Hamming Window:
hamming_window = sqrt(hamming(samples_per_frame));

%Estimate Noise Magnitude For First 'number_of_noise_frames_at_file_start' Frames:
[average_noise_fft_magnitude] = fft_calculate_average_power_spectrum_at_file_start(input_signal,number_of_noise_frames_at_file_start,samples_per_frame,hamming_window,fft_length);

%Reshape data vector to frames:
data_matrix = buffer(input_signal,length(hamming_window),number_of_common_samples_between_frames);
data_matrix = bsxfun(@times,data_matrix,hamming_window(:)); 
[number_of_samples_per_frame, number_of_frames] = size(data_matrix);

%Pre-allocate fft, abs(fft), angle(fft):
x_fft = fft(data_matrix,fft_length);
data_magnitude = abs(x_fft);
data_phase = angle(x_fft);

%Smooth magnitude spectrum between frames and across frames:
if flag_do_pre_smoothing_and_averaging==1
    %inter-frame weighted smoothing estimate:
    smoothing_filter = [0.9 0.1]; %what is this?
    data_magnitude_smoothed = zeros(size(data_magnitude));
    data_magnitude_smoothed = [[data_magnitude(1,1),data_magnitude(samples_per_frame-number_of_common_samples_between_frames,1:end-1)] ; data_magnitude];
    data_magnitude_smoothed = filter(smoothing_filter,1,data_magnitude_smoothed);
    data_magnitude_smoothed = data_magnitude_smoothed(2:end,:);
    
    %cross-frame weighted smoothing estimate: 
    weighted_smoothing_window = [0.09,0.25,0.32,0.25,0.09];
    data_magnitude_smoothed = reshape(conv(data_magnitude_smoothed(:),weighted_smoothing_window,'same'),size(data_magnitude_smoothed));
else
    data_magnitude_smoothed = data_magnitude;
end 

%Noise update during silent frames:
average_noise_fft_magnitude = repmat(average_noise_fft_magnitude,[1,number_of_frames]);
if flag_use_VAD==1
    [average_noise_fft_magnitude,VAD_over_time] = audio_VAD_and_update_noise_spectrum_only_time_domain(data_magnitude_smoothed,average_noise_fft_magnitude,samples_per_frame,number_of_common_samples_between_frames,number_of_frames);
end

%Calculte the segmental SNR in each band:
for bin_counter=1:number_of_frequency_bands
    for frame_counter=1:number_of_frames
        if bin_counter<number_of_frequency_bands
            segmented_frequency_binned_SNR(bin_counter,frame_counter) =  10*log10(norm(data_magnitude_smoothed(bin_start(bin_counter):bin_stop(bin_counter),frame_counter),2).^2/norm(average_noise_fft_magnitude(bin_start(bin_counter):bin_stop(bin_counter),frame_counter),2).^2);
        else
            segmented_frequency_binned_SNR(bin_counter,frame_counter) = 10*log10(norm(data_magnitude_smoothed(bin_start(bin_counter):fft_length/2+1,frame_counter),2).^2/norm(average_noise_fft_magnitude(bin_start(bin_counter):fft_length/2+1,frame_counter),2).^2);
        end
    end
end

%Calculate over-substraction factor:
low_SNR_logical_mat = segmented_frequency_binned_SNR<-5;
medium_SNR_logical_mat = (segmented_frequency_binned_SNR>=-5 & segmented_frequency_binned_SNR<=20);
high_SNR_logical_mat = segmented_frequency_binned_SNR>20;
over_substraction_factor_matrix = 4.75*low_SNR_logical_mat + (4-segmented_frequency_binned_SNR*3/20).*medium_SNR_logical_mat + 1*high_SNR_logical_mat;
over_substraction_factor_matrix = bsxfun(@times,over_substraction_factor_matrix,[1,2.5*ones(1,number_of_frequency_bands-2),1.5]');

%START SUBTRACTION PROCEDURE:
substracted_speech = zeros(fft_length/2+1,number_of_frames);
for band_counter=1:number_of_frequency_bands
    tic
    %current band indices:
    start = bin_start(band_counter);
    stop = bin_stop(band_counter); if band_counter==number_of_frequency_bands; stop = fft_length/2+1; end 

    %calculate over substracted speech:
    over_substraction_factor_vec = over_substraction_factor_matrix(band_counter,:);
    current_over_substraction_matrix = bsxfun(@times,average_noise_fft_magnitude(start:stop,:).^2,over_substraction_factor_vec(:)');
    current_substracted_speech = data_magnitude_smoothed(start:stop,:).^2 - current_over_substraction_matrix;
    
    %floor resultant spectrum:
    current_substracted_speech = max(spectral_floor*data_magnitude_smoothed(start:stop,:).^2,current_substracted_speech);
    
    %add a little bit of original spectrum for "naturality":
    current_substracted_speech = current_substracted_speech + (0.05 - 0.04*(bin_counter==number_of_frequency_bands))*data_magnitude_smoothed(start:stop,:).^2;
    
    %insert result into substracted speech total matrix:
    substracted_speech(start:stop,:) = current_substracted_speech; 
   toc 
end

%Reconstruct whole spectrum (it's symmetric around DC):
substracted_speech(fft_length/2+2:fft_length,:)=flipud(substracted_speech(2:fft_length/2,:));

%Multiply the whole frame fft with the phase information
y1_fft = sqrt(substracted_speech).*exp(1i*data_phase);

%Ensure a real signal (why not simplly real(ifft())?):
y1_fft(1,:) = real(y1_fft(1,:));
y1_fft(fft_length/2+1,:) = real(y1_fft(fft_length/2+1,:)); 

%IFFT: 
y1_ifft = ifft(y1_fft);
y1_real = real(y1_ifft);
 
%Overlap-Add:
y1(1:samples_per_frame)=y1_real(1:samples_per_frame,1);
start=samples_per_frame-number_of_overlap_samples_per_frame+1;
mid=start+number_of_overlap_samples_per_frame-1;
stop=start+samples_per_frame-1;
for frame_counter=2:number_of_frames
    y1(start:mid) = y1(start:mid)+y1_real(1:number_of_overlap_samples_per_frame,frame_counter)';
    y1(mid+1:stop) = y1_real(number_of_overlap_samples_per_frame+1:samples_per_frame,frame_counter);
    start = mid+1;
    mid=start+number_of_overlap_samples_per_frame-1;
    stop=start+samples_per_frame-1;
end
out=y1; 

sound(out(1:length(input_signal)),Fs);
wavwrite(out(1:length(input_signal)),Fs,16,output_file_name);


 



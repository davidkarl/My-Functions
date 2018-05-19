function [average_noise_fft_magnitude,VAD_over_time]=audio_VAD_and_update_noise_spectrum_only_time_domain(data_magnitude_smoothed,average_noise_fft_magnitude,samples_per_frame,number_of_common_samples_per_frame,number_of_frames)

%Define logicals:
SPEECH=1;	
SILENCE=0;

%Decide on smoothing:
a_smoothing = 0.9;

for frame_counter=1:number_of_frames
    %Build current spectrums and calculate current speech/silence judgement:
    current_data_frame_spectrum = data_magnitude_smoothed(:,frame_counter).^ 2;	
    current_noise_frame_spectrum = average_noise_fft_magnitude(:,frame_counter).^ 2;
    SNR_at_each_frequency = current_data_frame_spectrum./current_noise_frame_spectrum;
    rti = SNR_at_each_frequency - log10(SNR_at_each_frequency)-1; %WHAT IS THIS?
    judgevalue = mean(rti,1);
    judgevalue1((frame_counter-1)*number_of_common_samples_per_frame+1 : frame_counter*number_of_common_samples_per_frame) = judgevalue;
    
    %Decide if current frame is speech or silence and update average noise fft:
    if judgevalue>0.4
        VAD_over_time(1+(frame_counter-1)*samples_per_frame:frame_counter*samples_per_frame) = SPEECH;
    else
        VAD_over_time(1+(frame_counter-1)*samples_per_frame:frame_counter*samples_per_frame) = SILENCE;
        average_noise_fft_magnitude(:,frame_counter) = sqrt(a_smoothing*current_noise_frame_spectrum + (1-a_smoothing)*current_data_frame_spectrum);
    end
    
end



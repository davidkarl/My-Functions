%dsp book coding and decoding using AR model:


%dsp book coding speech based on an AR model:
% OUTPUT: array tab_cod(N,xx):
% tab_cod(N,1): energy in the block of signal
% tab_cod(N,2): pitch period
% tab_cod(N,3:12) AR coefficients (AR-ordv if voiced sound, AR-ordnv otherwise) or reflection coefficients
% each bock has 240 samples length(30ms) with 60 samples overlap
% load phrase;

input_signal = wavread('rodena blabla.wav');
input_signal = input_signal(:,1);
energy_threshold_for_pitch_detection = (std(input_signal))^2*0.1;

%AR model orders for voiced and non voiced sounds
AR_model_voiced_frame=20;
AR_model_unvoiced_frame=10;
max_total_AR_model_parameters = AR_model_voiced_frame+2;
phrase = input_signal-mean(input_signal);

%parameters:
samples_per_frame = 240; %block length
overlap_samples_per_frame = 60; %overlap
non_overlapping_samples_per_frame = samples_per_frame-overlap_samples_per_frame; 
number_of_frames = floor((length(phrase)-overlap_samples_per_frame)/non_overlapping_samples_per_frame); %Nb of blocks
signal_length_overflow_over_integer_number_of_blocks = rem( length(phrase) - overlap_samples_per_frame , non_overlapping_samples_per_frame );
phrase = phrase(1:length(phrase)-signal_length_overflow_over_integer_number_of_blocks);

%pitch detection parameters:
tmin=40;
tmax=150;
correlation_threshold_for_pitch_detection=0.7; %for pitch detection

%pitch detection vectors:
voice_or_unvoiced_frame_flags_vec=zeros(1,number_of_frames); %boolean voiced/non voiced
pitch_period_vec=zeros(1,number_of_frames); %pitch period
AR_estimated_coefficients_per_frame=zeros(number_of_frames,max_total_AR_model_parameters); %coeffts of the model

%detection voiced/non voiced:
sprintf('"voiced/non voiced" on %5.0f blocks',number_of_frames)

%LOOP over signal frames and decide whether voiced or not and detect pitch period:
tic
for k=1:number_of_frames
   start_index = (k-1)*non_overlapping_samples_per_frame + 1;
   stop_index = start_index + samples_per_frame - 1;
   current_signal_frame=phrase(start_index:stop_index); %analysis block
   
   %detect pitch:
   [voice_or_unvoiced_frame_flags_vec(k),pitch_period_vec(k)]=detectpitch(current_signal_frame,correlation_threshold_for_pitch_detection,tmin,tmax,energy_threshold_for_pitch_detection);
end
toc

%AR model:
sprintf('AR-model');
tic
pre_emphasized_signal=filter([1,-0.9375],1,phrase); %pre-emphasis
for k=2:(number_of_frames-1)
   
    %use detected pitch periods:
    if (voice_or_unvoiced_frame_flags_vec(k-1)==voice_or_unvoiced_frame_flags_vec(k+1)) %correction of errors of detection
       voice_or_unvoiced_frame_flags_vec(k)=voice_or_unvoiced_frame_flags_vec(k-1);
       if voice_or_unvoiced_frame_flags_vec(k)==1
          %voiced with pitch=mean
          pitch_period_vec(k)=floor((pitch_period_vec(k-1)+pitch_period_vec(k+1))/2);
       else
          %non voiced with pitch 0
          pitch_period_vec(k)=0;
       end
   end
   
   %analysis block:
   current_signal_frame = pre_emphasized_signal( (k-1)*non_overlapping_samples_per_frame+1 : (k-1)*non_overlapping_samples_per_frame+samples_per_frame );
   if voice_or_unvoiced_frame_flags_vec(k)==1
      [current_signal_frame_estimated_AR_model_parameters , estimated_AR_model_noise_variance] = ...
          xtoa(current_signal_frame,AR_model_voiced_frame);
      %coeff_refl=ai2ki(pcoeff); %reflection coeffts
      %tab_cod(k,3:NbParam)=coeff_refl; %coeffts
      AR_estimated_coefficients_per_frame(k,1)=estimated_AR_model_noise_variance;
      AR_estimated_coefficients_per_frame(k,2)=pitch_period_vec(k);
      AR_estimated_coefficients_per_frame(k,3:max_total_AR_model_parameters) = ...
          current_signal_frame_estimated_AR_model_parameters(2:AR_model_voiced_frame+1)';
      
   else
       [current_signal_frame_estimated_AR_model_parameters,estimated_AR_model_noise_variance]=xtoa(current_signal_frame,AR_model_unvoiced_frame);
       %coeff_refl=ai2ki(pcoeff); %reflection coeffts
       %tab_cod(k,3:NbParam)=coeff_refl;
       AR_estimated_coefficients_per_frame(k,1)=estimated_AR_model_noise_variance;
       AR_estimated_coefficients_per_frame(k,2)=0;
       AR_estimated_coefficients_per_frame(k,3:max_total_AR_model_parameters)=[current_signal_frame_estimated_AR_model_parameters(2:AR_model_unvoiced_frame+1)',zeros(1,AR_model_voiced_frame-AR_model_unvoiced_frame)];
   end
end
toc
sprintf('writing array in tab_cod.mat');
% save tab_cod AR_estimated_coefficients_per_frame



tic
% load tab_cod %tab_cod(nblocs,XX); 

%reconstructed signal parameters:
glottal_excitation_signal = eye(1,40); %glottal signal

%frame parameters for reconstruction:
overlap_reconstruction_samples_per_frame = samples_per_frame/3; %overlap reconstruction 1/3
reconstructed_signal_samples_per_frame = samples_per_frame+2*(overlap_reconstruction_samples_per_frame-overlap_samples_per_frame); %reconstructed block length
[number_of_frames , max_total_AR_model_parameters] = size(AR_estimated_coefficients_per_frame);
outsig=[];
finalsig=zeros(1,number_of_frames*non_overlapping_samples_per_frame+overlap_reconstruction_samples_per_frame);

%reconstruction window:
output_signal_reconstruction_window=[(1:overlap_reconstruction_samples_per_frame)/overlap_reconstruction_samples_per_frame, ones(1,samples_per_frame-2*overlap_samples_per_frame),(overlap_reconstruction_samples_per_frame:-1:1)/overlap_reconstruction_samples_per_frame];
overflow_of_integer_pitch_period_blocks_over_samples_per_frame=0;
glottal_excitation_signal_length=length(glottal_excitation_signal);
padded_number_of_samples_per_frame_for_filtering = reconstructed_signal_samples_per_frame+glottal_excitation_signal_length; %because of filtering
previous_frame_voice_or_unvoiced_flag=0;

for k=2:number_of_frames-1
   if AR_estimated_coefficients_per_frame(k,2)~=0 %voiced block
       if previous_frame_voice_or_unvoiced_flag==1 %the previous one is voiced
            %continuity of the input signal
            glottal_pulses_or_white_noise_to_feed_to_AR_filter=[previous_glottal_pulses_or_white_noise_to_feed_to_AR_filter(non_overlapping_samples_per_frame+1:padded_number_of_samples_per_frame_for_filtering),zeros(1,non_overlapping_samples_per_frame)];
            sample_to_begin_next_glottal_pulse_or_wn_construction = padded_number_of_samples_per_frame_for_filtering-non_overlapping_samples_per_frame+overflow_of_integer_pitch_period_blocks_over_samples_per_frame;
       else %the previous one is not voiced
           glottal_pulses_or_white_noise_to_feed_to_AR_filter=zeros(1,padded_number_of_samples_per_frame_for_filtering);
           sample_to_begin_next_glottal_pulse_or_wn_construction=0;
       end
       PitchPeriod=AR_estimated_coefficients_per_frame(k,2); %block pitch
       
       %generate glottal pulses for voiced speech:
       while sample_to_begin_next_glottal_pulse_or_wn_construction<reconstructed_signal_samples_per_frame
          glottal_pulses_or_white_noise_to_feed_to_AR_filter((sample_to_begin_next_glottal_pulse_or_wn_construction+1):(sample_to_begin_next_glottal_pulse_or_wn_construction+glottal_excitation_signal_length)) = glottal_excitation_signal;
          sample_to_begin_next_glottal_pulse_or_wn_construction=sample_to_begin_next_glottal_pulse_or_wn_construction+PitchPeriod;
       end 
       previous_frame_voice_or_unvoiced_flag=1;
       overflow_of_integer_pitch_period_blocks_over_samples_per_frame = sample_to_begin_next_glottal_pulse_or_wn_construction-padded_number_of_samples_per_frame_for_filtering;
       previous_glottal_pulses_or_white_noise_to_feed_to_AR_filter = glottal_pulses_or_white_noise_to_feed_to_AR_filter;
       glottal_pulses_or_white_noise_to_feed_to_AR_filter=glottal_pulses_or_white_noise_to_feed_to_AR_filter(1:reconstructed_signal_samples_per_frame);
       glottal_pulses_or_white_noise_to_feed_to_AR_filter=glottal_pulses_or_white_noise_to_feed_to_AR_filter/std(glottal_pulses_or_white_noise_to_feed_to_AR_filter); %normalization
       
   else %not voiced
       overflow_of_integer_pitch_period_blocks_over_samples_per_frame=0; 
       previous_frame_voice_or_unvoiced_flag=0; 
       glottal_pulses_or_white_noise_to_feed_to_AR_filter=randn(1,reconstructed_signal_samples_per_frame); %gaussian white noise
   end
   
   glottal_pulses_or_white_noise_to_feed_to_AR_filter=sqrt(AR_estimated_coefficients_per_frame(k,1))*glottal_pulses_or_white_noise_to_feed_to_AR_filter; %power
   %den=ki2ai(tab_cod(k,3:NbParam));
   den=[1,AR_estimated_coefficients_per_frame(k,3:max_total_AR_model_parameters)];
   outsig=filter(1,den,glottal_pulses_or_white_noise_to_feed_to_AR_filter);
   outsig=output_signal_reconstruction_window.*outsig;
   st=(k-1)*non_overlapping_samples_per_frame;
   
   %construction with an overlap:
   finalsig((st+1):(st+reconstructed_signal_samples_per_frame))=finalsig((st+1):(st+reconstructed_signal_samples_per_frame))+outsig;
   
end

finalsig=filter(1,[1,-0.9375],finalsig); %de emphasis
toc
soundsc(finalsig,44000);











function audio_log_MMSE(input_file_name,output_file_name,option_to_estimate_apriori_speech_absence_probability)
%         option  - method used to estimate the a priori probability of speech
%                   absence, P(Ho):
%                  1  - hard decision (Soon et al. [2])
%                  2  - soft decision (Soon et al. [2])
%                  3  - Malah et al.(1999) - ICASSP
%                  4  - Cohen (2002) [1]

       
%Input variables:
% input_file_name = '900m, reference loudness=69dBspl, conversation, laser power=1 , Fs=4935Hz, channel2.wav';
input_file_name = 'shirt_2mm_ver_200m_audioPM final demodulated audio170-3500[Hz]lsq smoother with deframing';
output_file_name = 'rodena_speaking_spectral_subtraction.wav';
option_to_estimate_apriori_speech_absence_probability = 4;

%Read input file:
[input_signal, Fs, bits] = wavread(input_file_name);
input_signal = input_signal(:,1);
input_signal = make_column(input_signal);
% input_signal = add_noise_of_certain_SNR(input_signal,3,1,0);

%Audio parameters:
frame_size_in_seconds = 0.02;   
samples_per_frame = make_even(floor(Fs*frame_size_in_seconds),1);
overlap_samples_per_frame = samples_per_frame/ 2;	
non_overlapping_samples_per_frame = samples_per_frame - overlap_samples_per_frame ;
hamming_window = make_column(hamming(samples_per_frame));
hamming_window = hamming_window * overlap_samples_per_frame / sum(hamming_window);
normalization_factor = ( hamming_window'* hamming_window)/ samples_per_frame;
FFT_size = samples_per_frame; 

%Get initial noise-only data samples and create a windowed matrix out of them:
number_of_initial_seconds_containing_only_noise = 0.5;
number_of_initial_samples_containing_only_noise = round(Fs*number_of_initial_seconds_containing_only_noise);
initial_samples_containing_only_noise = input_signal(1:number_of_initial_samples_containing_only_noise);
initial_noise_data_matrix = buffer(initial_samples_containing_only_noise,samples_per_frame,0);
initial_noise_data_matrix = bsxfun(@times,initial_noise_data_matrix,hamming_window);
average_noise_power_spectrum = mean( abs(fft(initial_noise_data_matrix,FFT_size)) , 2);
average_noise_power_spectrum = average_noise_power_spectrum.^2;
 
%Get only noisy speech after initial noise-only samples: 
noisy_speech = input_signal;
noisy_speech_number_of_samples = length(noisy_speech);
noisy_speech_number_of_frames = fix( (noisy_speech_number_of_samples-overlap_samples_per_frame) / non_overlapping_samples_per_frame);
initial_indices_of_frames_in_noisy_signal = 1 + (0:(noisy_speech_number_of_frames-1))*non_overlapping_samples_per_frame;
number_of_samples_resulting_from_buffering_input_signal = samples_per_frame + initial_indices_of_frames_in_noisy_signal(noisy_speech_number_of_frames) - 1;
if noisy_speech_number_of_samples < number_of_samples_resulting_from_buffering_input_signal
   %Zero pad signal to equate number of samples of buffered and original signal if necessary:
   noisy_speech(noisy_speech_number_of_samples+1 : samples_per_frame+initial_indices_of_frames_in_noisy_signal(noisy_speech_number_of_frames)-1) = 0;  
end

%Initialize helper variable which contains the overlapping samples of the previous frame:
enhanced_speech_overlapped_samples_from_previous_frame = zeros(overlap_samples_per_frame,1);

%Initialize final enhanced speech:
total_number_of_frames = floor(length(input_signal)/non_overlapping_samples_per_frame)-1;
final_enhanced_speech = zeros(total_number_of_frames*non_overlapping_samples_per_frame,1);

%Set parameter values:
smoothing_factor_in_noise_spectrum_update = 0.98;
smoothing_factor_in_apriori_SNR_update = 0.98;
VAD_threshold = 0.15; 
apriori_speech_absence_probability = 0.5 * ones(samples_per_frame,1);
apriori_speech_presence_probability = 1 - apriori_speech_absence_probability;
apriori_SNR_estimate_minimum = 10^(-25/10);
gain_function_minimum_assuming_speech_absence = 10^(-20/10);
xi_k_previous = zeros(samples_per_frame,1);

%Assign variables for Cohen's speech absence probability algorithm:
global xi_k_smoothed xi_k_smoothed_frequency_averaged_previous xi_k_maximum_ever
xi_k_smoothed = zeros(samples_per_frame/2+1,1); 
xi_k_smoothed_frequency_averaged_previous = 1000;  
xi_k_maximum_ever = 0; 

%Loop over noisy frames and enhance them:
for frame_counter = 1:total_number_of_frames
    tic
    %Get start and stop indices of current frame:
    start_index = initial_indices_of_frames_in_noisy_signal(frame_counter);
    stop_index = initial_indices_of_frames_in_noisy_signal(frame_counter) + samples_per_frame - 1;
    
    %Get noisy speech frame:
    current_frame = noisy_speech(start_index : stop_index);
    
    %Window current frame:
    current_frame = current_frame .* hamming_window;
    
    %Get current frame fft, magnitude, phase and spectrum:
    current_frame_fft = fft(current_frame,FFT_size);
    current_frame_fft_magnitude = abs(current_frame_fft);
    current_frame_fft_phase  = angle(current_frame_fft);
    current_frame_PSD = current_frame_fft_magnitude.^2;
     
    %aposteriori SNR estimate:
    aposteriori_SNR_estimate_per_frequency_current = min(current_frame_PSD ./ average_noise_power_spectrum,40);
    aposteriori_prime_current = max(aposteriori_SNR_estimate_per_frequency_current-1,0); 
    
    %apriori SNR estimate:
    if frame_counter==1
       apriori_SNR_estimate_per_frequency_using_previous_gain_function = 1; 
    else
       apriori_SNR_estimate_per_frequency_using_previous_gain_function = (gain_function_previous.^2) .* aposteriori_SNR_estimate_per_frequency_previous;  
    end
    
    %Smooth apriori SNR estimate and calculate respective aposteriori smoothed SNR estimate:
    apriori_SNR_estimate_smoothed = smoothing_factor_in_apriori_SNR_update * apriori_SNR_estimate_per_frequency_using_previous_gain_function ...
                                  + (1-smoothing_factor_in_apriori_SNR_update) * aposteriori_prime_current;
    apriori_SNR_estimate_smoothed = max(apriori_SNR_estimate_minimum,apriori_SNR_estimate_smoothed); 
    aposteriori_SNR_estimate_smoothed = apriori_SNR_estimate_smoothed + 1;
     
    %VAD decide whether speech is present and if not then update noise spectrum:
    log_likelihood_per_frequency = (aposteriori_SNR_estimate_per_frequency_current./aposteriori_SNR_estimate_smoothed).*apriori_SNR_estimate_smoothed - log(aposteriori_SNR_estimate_smoothed);
    vad_decision(frame_counter) = sum(log_likelihood_per_frequency) / samples_per_frame;   
    if (vad_decision(frame_counter) < VAD_threshold) 
        average_noise_power_spectrum = smoothing_factor_in_noise_spectrum_update*average_noise_power_spectrum + (1- smoothing_factor_in_noise_spectrum_update)*current_frame_PSD;
        vad_over_time( start_index : stop_index ) = 0;
    else
        vad_over_time( start_index : stop_index ) = 1;
    end
   
    %Assign simplifying variables for the following calculations:
    gamma_k = aposteriori_SNR_estimate_per_frequency_current;
    xi_k = apriori_SNR_estimate_smoothed;
    v_k = (xi_k./(1+xi_k)) .* gamma_k;
    
    %Assign simpliying noise and clean signal variance variables:
    Yk2_smoothed = aposteriori_SNR_estimate_per_frequency_current.*average_noise_power_spectrum;
    lambda_d = average_noise_power_spectrum;
    lambda_x = average_noise_power_spectrum .* apriori_SNR_estimate_smoothed;
    
    %Calculate conditional probabilities:
    probability_density_for_Yk_assuming_speech_is_absent = (1/pi./lambda_d) .* exp(-Yk2_smoothed./lambda_d);
    probability_density_for_Yk_assuming_speech_is_present = (1/pi./(lambda_d+lambda_x)) .* exp(-Yk2_smoothed./(lambda_d+lambda_x));
                         
    %Calculate apriori probability for speech absence and presence:
    [apriori_speech_absence_probability] = ...
        estimate_speech_absence_probability(apriori_speech_absence_probability,xi_k,xi_k_previous,gamma_k,option_to_estimate_apriori_speech_absence_probability);
    
    %Total probability for current noisy speech frame:
    total_probability_for_current_noisy_speech_frame = apriori_speech_presence_probability.*probability_density_for_Yk_assuming_speech_is_present + ...
                                                     + apriori_speech_absence_probability.*probability_density_for_Yk_assuming_speech_is_absent;
    aposteriori_speech_presence_probability = (apriori_speech_presence_probability.*probability_density_for_Yk_assuming_speech_is_present) ./ total_probability_for_current_noisy_speech_frame;
    aposteriori_speech_absence_probability = (apriori_speech_absence_probability.*probability_density_for_Yk_assuming_speech_is_absent) ./ total_probability_for_current_noisy_speech_frame;
    
    %Calculate gain function:
    gain_function_conditional_on_speech_presence = (xi_k./(1+xi_k)) .* exp(0.5*expint(v_k));
    gain_function_conditional_on_speech_absence  = gain_function_minimum_assuming_speech_absence;
    
    %Calculate final gain function:
    final_gain_function_current = (gain_function_conditional_on_speech_presence.^aposteriori_speech_presence_probability) .* (gain_function_conditional_on_speech_absence.^aposteriori_speech_absence_probability);
    
    %Calculate Enhanced speech frame: 
    enhanced_frame_current = real(ifft( current_frame_fft_magnitude .* final_gain_function_current .* exp(1i*current_frame_fft_phase), FFT_size));

    %Overlap-Add:    
    first_overlapping_part_of_enhanced_signal = enhanced_frame_current(1:overlap_samples_per_frame) + enhanced_speech_overlapped_samples_from_previous_frame;
    second_nonoverlapping_part_of_enhanced_signal = enhanced_frame_current(overlap_samples_per_frame+1 : samples_per_frame);
    current_final_enhanced_speech_frame = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
    final_enhanced_speech(start_index:stop_index) = [first_overlapping_part_of_enhanced_signal ; second_nonoverlapping_part_of_enhanced_signal];
     
    %Remember overlapping part for next overlap-add:
    enhanced_speech_overlapped_samples_from_previous_frame = enhanced_frame_current( samples_per_frame-overlap_samples_per_frame+1 : samples_per_frame);
    
    %Remember current wiener filter gain function for later apriori SNR estimation: 
    gain_function_previous = final_gain_function_current; 
    xi_k_previous = xi_k;
    aposteriori_SNR_estimate_per_frequency_previous = aposteriori_SNR_estimate_per_frequency_current;
    toc
end 
   
%Write down final enhanced speech: 
sound(final_enhanced_speech,Fs);
wavwrite(final_enhanced_speech,Fs,16,output_file_name);

function [qk] = estimate_speech_absence_probability(qk,xi_k,xi_k_previous,gamma_k,type)

% Returns a priori probability of speech absence, P(Ho):

global xi_k_smoothed xi_k_smoothed_frequency_averaged_previous xi_k_maximum_ever

if type ==1
%Hard Decision:
    
    %Decide on smoothing factor for speech absence probability undate:
    smoothing_factor_in_speech_absence_probability_update = 0.9;

    %Decide on current speech presence in each frequency:
    speech_absence_probability_current = ones(length(xi_k),1);
    speech_absence_probability_current(exp(-xi_k).*besseli(0,2*sqrt(gamma_k.*xi_k)) > 1) = 0;
    
    %Update qk:
    qk = smoothing_factor_in_speech_absence_probability_update * qk ...
      + (1-smoothing_factor_in_speech_absence_probability_update) * speech_absence_probability_current;

elseif type==2
%Soft Decision:

    %Decide on smoothing factor for speech absence probability undate:
    smoothing_factor_in_speech_absence_probability_update = 0.9;
    
    %Decide on current speech presence in each frequency:
    speech_absence_probability_current = min(1 ./ (1 + exp(-xi_k).*besseli(0,2*sqrt(gamma_k.*xi_k))) , 1);
    
    %Update qk:
    qk = smoothing_factor_in_speech_absence_probability_update * qk ...
      + (1-smoothing_factor_in_speech_absence_probability_update) * speech_absence_probability_current;
    
elseif type==3
%Malah:
    
    %Decide on smoothing factor for speech absence probability undate:
    smoothing_factor_in_speech_absence_probability_update = 0.95;
    
    %VAD detector:
    VAD_threshold = 2;
    VAD_decision = mean(gamma_k(1:floor(length(gamma_k)/2)));
    
    %Update only if VAD gives that speech is present:
    if VAD_decision > VAD_threshold
        
      %decide on gamma_k threshold:
      gamma_k_th = 0.8;
      speech_absence_probability_current = ones(length(xi_k),1);
      speech_absence_probability_current( gamma_k > gamma_k_th ) = 0;
      
      %Update qk:
      qk = smoothing_factor_in_speech_absence_probability_update * qk ...
         +(1-smoothing_factor_in_speech_absence_probability_update) * speech_absence_probability_current;
    end
    
elseif type==4
%Cohen:
    
    %Decide on smoothing factor for speech absence probability undate:
    smoothing_factor_in_xi_k_update = 0.7;
    
    %Smooth xi_k between frames:
    one_sided_frequency_vec_length = length(qk)/2+1;
    xi_k_smoothed = smoothing_factor_in_xi_k_update * xi_k_smoothed ...
                  + (1-smoothing_factor_in_xi_k_update)*xi_k_previous(1:one_sided_frequency_vec_length);
    
    %Decide on minimum and maximum parameters for xi_k smoothing for later use:
    xi_k_smoothed_minimum = 0.1; 
    xi_k_smoothed_maximum = 0.3162;
    xi_k_averaged_all_frequencies_minimum = 1; 
    xi_k_averaged_all_frequencies_maximum = 10;
      
    %Estimate speech presence probability inside a small local area on the frequency vec:
    xi_k_smoothed_small_local_average = smooth_vector_over_N_neighbors(xi_k_smoothed,1);
    xi_k_small_local_average_zero_offset_linear_in_log_space = log10(xi_k_smoothed_small_local_average/xi_k_smoothed_minimum) / log10(xi_k_smoothed_maximum/xi_k_smoothed_minimum);
    indices_where_small_local_speech_presence_probability_is_small = (xi_k_smoothed_small_local_average < xi_k_smoothed_minimum);
    indices_where_small_local_speech_presence_probability_is_medium = ((xi_k_smoothed_small_local_average > xi_k_smoothed_minimum) & (xi_k_smoothed_small_local_average < xi_k_smoothed_maximum));
    indices_where_small_local_speech_presence_probability_is_large = (xi_k_smoothed_small_local_average > xi_k_smoothed_maximum);
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_small) = 0;
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_medium) = xi_k_small_local_average_zero_offset_linear_in_log_space(indices_where_small_local_speech_presence_probability_is_medium);
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_large) = 1;
    
    %Estimate speech presence probability inside a large local area on the frequency vec:
    xi_k_smoothed_large_local_average = smooth_vector_over_N_neighbors(xi_k_smoothed,15);
    xi_k_large_local_average_zero_offset_linear_in_log_space = log10(xi_k_smoothed_large_local_average/xi_k_smoothed_minimum) / log10(xi_k_smoothed_maximum/xi_k_smoothed_minimum);
    indices_where_large_local_speech_presence_probability_is_small = (xi_k_smoothed_large_local_average < xi_k_smoothed_minimum);
    indices_where_large_local_speech_presence_probability_is_medium = ((xi_k_smoothed_large_local_average > xi_k_smoothed_minimum) & (xi_k_smoothed_large_local_average < xi_k_smoothed_maximum));
    indices_where_large_local_speech_presence_probability_is_large = (xi_k_smoothed_large_local_average > xi_k_smoothed_maximum);
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_small) = 0;
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_medium) = xi_k_large_local_average_zero_offset_linear_in_log_space(indices_where_large_local_speech_presence_probability_is_medium);
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_large) = 1;
    
    %Estimate speech presence probability over all frequency vec:
    xi_k_smoothed_averaged_over_all_frequencies_current = mean(xi_k_smoothed);
    if  xi_k_smoothed_averaged_over_all_frequencies_current>xi_k_smoothed_minimum
        %If xi_k_smoothed_averaged is larger then some minimum then check and decide
        %as to what to assign to P_frame_average_current:
        
        if xi_k_smoothed_averaged_over_all_frequencies_current>xi_k_smoothed_frequency_averaged_previous
            %If xi_k_smoothed_averaged got raised from last frame the P_frame_average_current = 1:
            P_speech_presence_probability_over_all_frequencies = 1;
            xi_k_maximum_ever = min(max(xi_k_smoothed_averaged_over_all_frequencies_current,xi_k_averaged_all_frequencies_minimum),xi_k_averaged_all_frequencies_maximum);
            
        else
            %If xi_k_smoothed_averaged is lower then last frame then check where the value lies:
            if xi_k_smoothed_averaged_over_all_frequencies_current <= xi_k_maximum_ever*xi_k_smoothed_minimum 
                P_speech_presence_probability_over_all_frequencies = 0;
            elseif xi_k_smoothed_averaged_over_all_frequencies_current >= xi_k_maximum_ever*xi_k_smoothed_maximum 
                P_speech_presence_probability_over_all_frequencies = 1;
            else
                P_speech_presence_probability_over_all_frequencies = log10(xi_k_smoothed_averaged_over_all_frequencies_current/xi_k_maximum_ever/xi_k_smoothed_minimum) / log10(xi_k_smoothed_maximum/xi_k_smoothed_minimum);
            end
            
        end
        
    else
        P_speech_presence_probability_over_all_frequencies = 0;
    end
    
    %Estimate final speech presence probability probability using Cohen's method:
    qk = 1 - P_speech_presence_probability_in_small_local_frequency_range.*P_speech_presence_probability_in_large_local_frequency_range*P_speech_presence_probability_over_all_frequencies;
    qk = min(0.95,qk);
    qk = [qk , flipud(qk(2:one_sided_frequency_vec_length-1))];
    qk = qk(:);
    
    %Keep track of xi_k_smoothed_frequency_averaged for next evaluation phase:
    xi_k_smoothed_frequency_averaged_previous = xi_k_smoothed_averaged_over_all_frequencies_current;  
end
    



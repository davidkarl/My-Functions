function parameters = noise_estimation_IDIMCRA(current_frame_ps,parameters)
%         parameters = struct('n',2,'len',len_val,'noise_cap',ns_ps,'noise_tild',ns_ps,'gamma',ones(len_val,1),'Sf',Sf_val,...
%             'Smin',Sf_val,'S',Sf_val,'S_tild',Sf_val,'GH1',ones(len_val,1),'Smin_tild',Sf_val,'Smin_sw',Sf_val,'Smin_sw_tild',Sf_val,...
%             'stored_min',max(ns_ps)*ones(len_val,U_val),'stored_min_tild',max(ns_ps)*ones(len_val,U_val)','u1',1,'u2',1,'j',2,...
%             'alpha_d',0.85,'alpha_s',0.9,'U',8,'V',15,'Bmin',1.66,'gamma0',4.6,'gamma1',3,'psi0',1.67,'alpha',0.92,'beta',1.47,...
%             'b',b_val,'Sf_tild',Sf_tild_val);

global current_frame_size frame_counter spectrum_minimum_buildup_counter number_of_spectral_minimum_buildup_blocks_to_remember
global number_of_frames_to_buildup_spectrum_minimum total_number_of_frames_where_minimum_is_searched
global u1 u2 u3 u4
global raw_frequency_smoothed_ps_time_smoothed raw_frequency_smoothed_ps_time_smoothed2
global raw_frequency_smoothed_primarily_noise_ps_time_smoothed raw_frequency_smoothed_primarily_noise_ps_time_smoothed2
global apriori_SNR_smoothing_factor raw_frequency_smoothed_ps_time_smoothing_factor final_noise_speech_activity_probability_smoothing_factor
global current_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold smoothed_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold
global current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold
global small_local_frequency_window_smoother large_local_frequency_window_smoother
global raw_ps_to_smoothed_noise_ps_aposteriori_SNR_previous 
global final_noise_ps_after_beta_correction_previous final_noise_ps_before_beta_correction_previous
global spectral_magnitude_gain_function_previous Bmin_minimum_ps_correction_factor final_noise_correction_factor
global maximum_raw_aposteriori_SNR
global flag_gain_method bayesian_cost_function_p_power minimum_attenuation_correction_factor
global smoothed_apriori_SNR_frequency_averaged_previous smoothed_apriori_SNR_constrained_peak
global flag_use_omlsa_speech_absence_or_minima_controled flag_omlsa_use_small_local_indicator flag_omlsa_use_large_local_indicator
global flag_omlsa_use_global_indicator q_apriori_speech_absence_probability_maximum p_speech_presence_probability_maximum
global flag_smooth_speech_presence_probability flag_smooth_apriori_speech_absence_probability
global number_of_bins_to_smooth_speech_presence_probability flag_smooth_speech_presence_linear_logarithmic_or_raised_decay raised_decay_dB_per_bin
global indices_to_look_for_low_frequencies_speech_activity_min indices_to_look_for_low_frequencies_speech_activity_max
global P_min_low_frequencies_mean_speech_probability_threshold P_min less_smoothed_raw_ps xi_smoothing_factor smoothed_xi_estimate flag_remove_interfering_tonals
global stored_raw_window_averaged_ps_smoothed_min stored_raw_window_averaged_ps_smoothed_min2 
global stored_noise_window_averaged_ps_smoothed_min stored_noise_window_averaged_ps_smoothed_min2
global raw_frequency_smoothed_ps_time_smoothed_min raw_frequency_smoothed_ps_time_smoothed_min2
global raw_frequency_smoothed_ps_time_smoothed_min_sw raw_frequency_smoothed_ps_time_smoothed_min_sw2
global raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2
global raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw2




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UPDATE AND CALCULATE THINGS INVOLVING RAW (SIGNAL ONLY) POWER SPECTRUM:
%Calculate current raw ps to smoothed noise ps APOSTERIORI SNR (gamma): 
raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current = current_frame_ps ./ final_noise_ps_after_beta_correction_previous;

%limit APOSTERIORI SNR:
raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current = min(raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current,maximum_raw_aposteriori_SNR);

%Calculate APRIORI SNR estimate (epsilon):
previous_smoothed_apriori_SNR_estimate = spectral_magnitude_gain_function_previous.^2 .* raw_ps_to_smoothed_noise_ps_aposteriori_SNR_previous;
current_raw_apriori_SNR_estimate_using_aposteriori_SNR_estimate = max(raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current-1,0); 
current_apriori_SNR_smoothed_estimate = apriori_SNR_smoothing_factor * previous_smoothed_apriori_SNR_estimate ...
    + (1-apriori_SNR_smoothing_factor) * current_raw_apriori_SNR_estimate_using_aposteriori_SNR_estimate;


%Compute current v_k:
v_k = (current_apriori_SNR_smoothed_estimate./(1+current_apriori_SNR_smoothed_estimate)) .* raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current;

%Compute Spectral Gain According to Desired Algorithm:
if flag_gain_method==1
    %log-MMSE:
    spectral_magnitude_gain_function_current = (current_apriori_SNR_smoothed_estimate./(1+current_apriori_SNR_smoothed_estimate)) .*exp(1/2*expint(v_k));
elseif flag_gain_method==2
    %weighted distortion ratio:
    if bayesian_cost_function_p_power == -1
        spectral_magnitude_gain_function_current =  gamma((bayesian_cost_function_p_power+3)/2)/gamma(bayesian_cost_function_p_power/2+1) .* (sqrt(v_k)./raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current) ./ exp(-v_k/2) ./ besseli(0,v_k/2);
    else
        gain_function_numerator = gamma((bayesian_cost_function_p_power+3)/2)/gamma(bayesian_cost_function_p_power/2+1) .* sqrt(v_k) .* confhyperg(-(bayesian_cost_function_p_power+1)/2,1,-v_k,100);
        gain_function_denominator = raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current .* confhyperg(-bayesian_cost_function_p_power/2,1,-v_k,100);
        spectral_magnitude_gain_function_current = gain_function_numerator ./ gain_function_denominator;
    end
elseif flag_gain_method==3
    %weighted likelihood ratio:
    gain_function_numerator = sqrt(gamma((bayesian_cost_function_p_power+3)/2)/gamma((bayesian_cost_function_p_power+1)/2)) .* sqrt(v_k.*confhyperg(-(bayesian_cost_function_p_power+1)/2,1,-v_k,100));
    gain_function_denominator = raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current .* sqrt(confhyperg(-(bayesian_cost_function_p_power-1)/2,1,-v_k,100));
    spectral_magnitude_gain_function_current = gain_function_numerator ./ gain_function_denominator;
else
    %Cosh distortion:
    gain_function_numerator = sqrt(gamma((bayesian_cost_function_p_power+3)/2)/gamma((bayesian_cost_function_p_power+1)/2)) .* sqrt(v_k.*confhyperg(-(bayesian_cost_function_p_power+1)/2,1,-v_k,100));
    gain_function_denominator = raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current .* sqrt(confhyperg(-(bayesian_cost_function_p_power-1)/2,1,-v_k,100));
    spectral_magnitude_gain_function_current = gain_function_numerator ./ gain_function_denominator;
end 
indices_where_vk_is_large = (v_k>5);
spectral_magnitude_gain_function_current(indices_where_vk_is_large) = (current_apriori_SNR_smoothed_estimate(indices_where_vk_is_large)./(1+current_apriori_SNR_smoothed_estimate(indices_where_vk_is_large))) ;

    
%Calculate window-averaged RAW power spectrum (SIGNAL+NOISE):
%SMALL WINDOW:
[raw_frequency_smoothed_ps] = conv_without_end_effects(current_frame_ps,small_local_frequency_window_smoother);
% [raw_frequency_smoothed_ps] = smooth_vec_by_decay(current_frame_ps,3,length(small_local_frequency_window_smoother));
%LARGE WINDOW: 
[raw_frequency_smoothed_ps2] = conv_without_end_effects(current_frame_ps,large_local_frequency_window_smoother);


%Smooth raw frequency smoothed ps in time domain:
%Small window:
raw_frequency_smoothed_ps_time_smoothed = raw_frequency_smoothed_ps_time_smoothing_factor*raw_frequency_smoothed_ps_time_smoothed ... 
                                + (1-raw_frequency_smoothed_ps_time_smoothing_factor)*raw_frequency_smoothed_ps;
%Large window:
raw_frequency_smoothed_ps_time_smoothed2 = raw_frequency_smoothed_ps_time_smoothing_factor*raw_frequency_smoothed_ps_time_smoothed2 ... 
                                + (1-raw_frequency_smoothed_ps_time_smoothing_factor)*raw_frequency_smoothed_ps2;

%Calculate ps to smoothed Pmin aposteriori SNR (gamma_min) and smooth ps to smooth ps minimum (psi_min):
%Use sorting object to track min and median of raw frequency and time smoothed ps:
%SMALL WINDOW:
raw_frequency_smoothed_ps_time_smoothed_min = min(raw_frequency_smoothed_ps_time_smoothed_min,raw_frequency_smoothed_ps_time_smoothed);
raw_frequency_smoothed_ps_time_smoothed_min_sw = min(raw_frequency_smoothed_ps_time_smoothed_min_sw,raw_frequency_smoothed_ps_time_smoothed);
current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min = current_frame_ps ./ (Bmin_minimum_ps_correction_factor*max(10^-10,raw_frequency_smoothed_ps_time_smoothed_min));
smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min = raw_frequency_smoothed_ps_time_smoothed ./ (Bmin_minimum_ps_correction_factor*max(10^-10,raw_frequency_smoothed_ps_time_smoothed_min));

%LARGE WINDOW:
raw_frequency_smoothed_ps_time_smoothed_min2 = min(raw_frequency_smoothed_ps_time_smoothed_min2,raw_frequency_smoothed_ps_time_smoothed2);
raw_frequency_smoothed_ps_time_smoothed_min_sw2 = min(raw_frequency_smoothed_ps_time_smoothed_min_sw2,raw_frequency_smoothed_ps_time_smoothed2);
current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min2 = current_frame_ps ./ (Bmin_minimum_ps_correction_factor*max(10^-10,raw_frequency_smoothed_ps_time_smoothed_min2));
smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min2 = raw_frequency_smoothed_ps_time_smoothed2 ./ (Bmin_minimum_ps_correction_factor*max(10^-10,raw_frequency_smoothed_ps_time_smoothed_min2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UPDATE AND CALCULATE THINGS INVOLVING PRIMARILY NOISE POWER SPECTRUM:
%%%% SMALL WINDOW:
%Build logical mask which has ones where speech is deemed absent (and we might need to update ps around it):
logical_mask_where_speech_is_absent = zeros(current_frame_size,1);
logical_mask_where_speech_is_absent(...
    current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min < current_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold ... 
    & smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min < smoothed_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold) = 1;  

 
%Update Sf_tilde to update S_tilde in frequencies where in their vicinity were found to contain only noise:
[raw_frequency_smoothed_primarily_noise_ps] = average_over_indicators_and_smoothing_window(current_frame_ps,logical_mask_where_speech_is_absent,small_local_frequency_window_smoother,raw_frequency_smoothed_primarily_noise_ps_time_smoothed);
raw_frequency_smoothed_primarily_noise_ps_time_smoothed = raw_frequency_smoothed_ps_time_smoothing_factor*raw_frequency_smoothed_primarily_noise_ps_time_smoothed ...
                                                + (1-raw_frequency_smoothed_ps_time_smoothing_factor)*raw_frequency_smoothed_primarily_noise_ps;

%Calculate noise ps to smoothed P_noise_min aposteriori SNR (gamma_min_tilde) 
%and smooth noise ps to smooth noise ps minimum (psi_min_tilde):
%Use sorting object to track min and median of raw frequency and time smoothed mainly noise ps:
raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min = min(raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min,max(10^-10,raw_frequency_smoothed_primarily_noise_ps_time_smoothed));
raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw = min(raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw,max(10^-10,raw_frequency_smoothed_primarily_noise_ps_time_smoothed));

current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR = current_frame_ps ./ (Bmin_minimum_ps_correction_factor*max(10^-10,raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min));
smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR = raw_frequency_smoothed_primarily_noise_ps_time_smoothed ./ (Bmin_minimum_ps_correction_factor*max(10^-10,raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min));


%%%% LARGE WINDOW:
%Build logical mask which has ones where speech is deemed absent (and we might need to update ps around it):
logical_mask_where_speech_is_absent2 = zeros(current_frame_size,1);
logical_mask_where_speech_is_absent2(...
    current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min2 < current_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold ... 
    & smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min2 < smoothed_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold) = 1;  

 
%Update Sf_tilde to update S_tilde in frequencies where in their vicinity were found to contain only noise:
[raw_frequency_smoothed_primarily_noise_ps2] = average_over_indicators_and_smoothing_window(current_frame_ps,logical_mask_where_speech_is_absent2,large_local_frequency_window_smoother,raw_frequency_smoothed_primarily_noise_ps_time_smoothed);
raw_frequency_smoothed_primarily_noise_ps_time_smoothed2 = raw_frequency_smoothed_ps_time_smoothing_factor*raw_frequency_smoothed_primarily_noise_ps_time_smoothed2 ...
                                                + (1-raw_frequency_smoothed_ps_time_smoothing_factor)*raw_frequency_smoothed_primarily_noise_ps2;

%Calculate noise ps to smoothed P_noise_min aposteriori SNR (gamma_min_tilde) 
%and smooth noise ps to smooth noise ps minimum (psi_min_tilde):
%Use sorting object to track min and median of raw frequency and time smoothed mainly noise ps:
raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2 = min(raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2,raw_frequency_smoothed_primarily_noise_ps_time_smoothed2);
raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw2 = min(raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw2,raw_frequency_smoothed_primarily_noise_ps_time_smoothed2);

current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2 = current_frame_ps ./ (Bmin_minimum_ps_correction_factor*max(raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2,10^-10));
smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2 = raw_frequency_smoothed_primarily_noise_ps_time_smoothed2 ./ (Bmin_minimum_ps_correction_factor*max(raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2,10^-10));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v%%%%%%%%%%%%%%%%%%%%
%CALCULATE APPROXIMATE A-PRIORI SPEECH ABSENCE PROBABILITY AND CONDITIONAL SPEECH PRESENCE PROBABILITY:
q_apriori_speech_absence_probability = zeros(current_frame_size,1);
q_apriori_speech_absence_probability2 = zeros(current_frame_size,1);

if flag_use_omlsa_speech_absence_or_minima_controled==0
    %Use omlsa method (apriori speech absence through apriori SPEECH PRESENCE estimation):
    
    %Initialize P speech presence probability contributions:
    P_speech_presence_probability_in_small_local_frequency_range = zeros(size(current_apriori_SNR_smoothed_estimate));
    P_speech_presence_probability_in_large_local_frequency_range = zeros(size(current_apriori_SNR_smoothed_estimate));
    P_contribution_from_apriori_SNR_global_sum = zeros(size(current_apriori_SNR_smoothed_estimate));
    
    %Decide on minimum and maximum parameters for xi_k smoothing for later use:
    smoothed_apriori_SNR_minimum = 0.1;
    smoothed_apriori_SNR_maximum = 0.3162;
    smoothed_apriori_SNR_global_sum_minimum = 1;
    smoothed_apriori_SNR_global_sum_maximum = 10;
    
    %Estimate speech presence probability inside a small local area on the frequency vec:
    smoothed_xi_estimate = xi_smoothing_factor*smoothed_xi_estimate + (1-xi_smoothing_factor)*current_apriori_SNR_smoothed_estimate;
    smoothed_apriori_SNR_small_local_average = conv_without_end_effects(smoothed_xi_estimate,small_local_frequency_window_smoother);
    P_contribution_by_small_local_sum_of_apriori_SNR = P_min + (1-P_min)*log10(smoothed_apriori_SNR_small_local_average/smoothed_apriori_SNR_minimum) / log10(smoothed_apriori_SNR_maximum/smoothed_apriori_SNR_minimum);
    indices_where_small_local_speech_presence_probability_is_small = (smoothed_apriori_SNR_small_local_average < smoothed_apriori_SNR_minimum);
    indices_where_small_local_speech_presence_probability_is_medium = ((smoothed_apriori_SNR_small_local_average > smoothed_apriori_SNR_minimum) & (smoothed_apriori_SNR_small_local_average < smoothed_apriori_SNR_maximum));
    indices_where_small_local_speech_presence_probability_is_large = (smoothed_apriori_SNR_small_local_average > smoothed_apriori_SNR_maximum);
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_small) = P_min;
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_medium) = P_contribution_by_small_local_sum_of_apriori_SNR(indices_where_small_local_speech_presence_probability_is_medium);
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_large) = 1;
    
    %Estimate speech presence probability inside a large local area on the frequency vec:
    xi_k_smoothed_large_local_average = conv_without_end_effects(smoothed_xi_estimate,large_local_frequency_window_smoother);
    P_contribution_by_large_local_sum_of_apriori_SNR = P_min + (1-P_min)*log10(xi_k_smoothed_large_local_average/smoothed_apriori_SNR_minimum) / log10(smoothed_apriori_SNR_maximum/smoothed_apriori_SNR_minimum);
    indices_where_large_local_speech_presence_probability_is_small = (xi_k_smoothed_large_local_average < smoothed_apriori_SNR_minimum);
    indices_where_large_local_speech_presence_probability_is_medium = ((xi_k_smoothed_large_local_average > smoothed_apriori_SNR_minimum) & (xi_k_smoothed_large_local_average < smoothed_apriori_SNR_maximum));
    indices_where_large_local_speech_presence_probability_is_large = (xi_k_smoothed_large_local_average > smoothed_apriori_SNR_maximum);
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_small) = P_min;
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_medium) = P_contribution_by_large_local_sum_of_apriori_SNR(indices_where_large_local_speech_presence_probability_is_medium);
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_large) = 1;
    
    %Estimate speech presence probability over all frequency vec:
    smoothed_apriori_SNR_averaged_over_all_frequencies_current = mean(smoothed_xi_estimate);
    if  smoothed_apriori_SNR_averaged_over_all_frequencies_current>smoothed_apriori_SNR_minimum
        %If xi_k_smoothed_averaged is larger then some minimum then check and decide
        %as to what to assign to P_frame_average_current:
        if smoothed_apriori_SNR_averaged_over_all_frequencies_current>smoothed_apriori_SNR_frequency_averaged_previous
            %If xi_k_smoothed_averaged got raised from last frame the P_frame_average_current = 1:
            P_contribution_from_apriori_SNR_global_sum = 1;
            smoothed_apriori_SNR_constrained_peak = min(max(smoothed_apriori_SNR_averaged_over_all_frequencies_current,smoothed_apriori_SNR_global_sum_minimum),smoothed_apriori_SNR_global_sum_maximum);
        else
            %If xi_k_smoothed_averaged is lower then last frame then check where the value lies:
            if smoothed_apriori_SNR_averaged_over_all_frequencies_current <= smoothed_apriori_SNR_constrained_peak*smoothed_apriori_SNR_minimum
                P_contribution_from_apriori_SNR_global_sum = P_min;
            elseif smoothed_apriori_SNR_averaged_over_all_frequencies_current >= smoothed_apriori_SNR_constrained_peak*smoothed_apriori_SNR_maximum
                P_contribution_from_apriori_SNR_global_sum = 1;
            else
                P_contribution_from_apriori_SNR_global_sum = P_min + (1-P_min)*log10(smoothed_apriori_SNR_averaged_over_all_frequencies_current/smoothed_apriori_SNR_constrained_peak/smoothed_apriori_SNR_minimum) / log10(smoothed_apriori_SNR_maximum/smoothed_apriori_SNR_minimum);
            end
        end
    else
        P_contribution_from_apriori_SNR_global_sum = P_min;
    end
    
    
    %Look at low frequencies where speech is likely to be present and if
    %there's low energy there then reset speech probabilities there:
    P_global_low_frequencies_mean_speech_probability = ...
        mean(P_speech_presence_probability_in_small_local_frequency_range(3:indices_to_look_for_low_frequencies_speech_activity_max)); 
    if P_global_low_frequencies_mean_speech_probability < P_min_low_frequencies_mean_speech_probability_threshold
        P_speech_presence_probability_in_small_local_frequency_range(indices_to_look_for_low_frequencies_speech_activity_min:indices_to_look_for_low_frequencies_speech_activity_max) ...
            = P_min;
    end
    
    %Remove interfering tonals if wanted:
    if flag_remove_interfering_tonals==1
        if (P_global_low_frequencies_mean_speech_probability<0.5)
            indices_with_interfering_tonals=find( final_noise_ps_before_beta_correction_previous(8:end-8) > ...
                2.5*(final_noise_ps_before_beta_correction_previous(10:end-6)+final_noise_ps_before_beta_correction_previous(6:end-10)) );
            P_speech_presence_probability_in_small_local_frequency_range([indices_with_interfering_tonals+6;indices_with_interfering_tonals+7;indices_with_interfering_tonals+8]) = P_min;   
        end
    end   
    
    
    %Construct final P_apriori_speech_presence_probability:
    normalizing_probability_power_for_geometric_mean = 0;
    flag_normalize_P_according_to_geometric_mean = 0;
    P_apriori_speech_presence_probability = 1;
    if flag_omlsa_use_small_local_indicator==1
        normalizing_probability_power_for_geometric_mean = normalizing_probability_power_for_geometric_mean + 1;
        P_apriori_speech_presence_probability = P_apriori_speech_presence_probability.*P_speech_presence_probability_in_small_local_frequency_range;
    end
    if flag_omlsa_use_large_local_indicator==1
        normalizing_probability_power_for_geometric_mean = normalizing_probability_power_for_geometric_mean + 1;
        P_apriori_speech_presence_probability = P_apriori_speech_presence_probability.*P_speech_presence_probability_in_large_local_frequency_range;
    end
    if flag_omlsa_use_global_indicator==1
        normalizing_probability_power_for_geometric_mean = normalizing_probability_power_for_geometric_mean + 1;
        P_apriori_speech_presence_probability = P_apriori_speech_presence_probability.*P_contribution_from_apriori_SNR_global_sum;
    end
    if flag_normalize_P_according_to_geometric_mean==1
        P_apriori_speech_presence_probability = P_apriori_speech_presence_probability.^(1/normalizing_probability_power_for_geometric_mean);
    end
    
    %Estimate final speech presence probability probability using Cohen's method:
    q_apriori_speech_absence_probability = 1 - P_apriori_speech_presence_probability;

    %Assign indices where speech probability is to updated to all:
    indices_where_speech_is_deemed_probably_present = (q_apriori_speech_absence_probability<0.9);
    
    
    
    
    
    
    
    
else
    
    %Use Original IMCRA method (direct estimation of q apriori speech ABSENCE):
    
    %SMALL WINDOW:
    indices_where_speech_is_deemed_absolutely_absent = ...
        (current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR<=1 & ...
        smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR < smoothed_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold);
    indices_where_speech_is_deemed_probably_present = not(indices_where_speech_is_deemed_absolutely_absent);
    indices_where_speech_is_deemed_probably_absent = ...
        (current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR>1 & ...
        current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR<current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold & ...
        smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR<smoothed_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold);
    %put 1s in speech absence probability where i found speech to be absent with very high probability:
    if (~isempty(indices_where_speech_is_deemed_absolutely_absent))
        q_apriori_speech_absence_probability(indices_where_speech_is_deemed_absolutely_absent) = 1;
    end
    %initialize 0 in all places where speech is not deemed absolutely absent:
    if (~isempty(indices_where_speech_is_deemed_probably_present))
        q_apriori_speech_absence_probability(indices_where_speech_is_deemed_probably_present) = 0;
    end
    %use linear soft decision where speech is deemed only probably absent:
    if (~isempty(indices_where_speech_is_deemed_probably_absent))
        q_apriori_speech_absence_probability(indices_where_speech_is_deemed_probably_absent) = max( (current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold-current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR(indices_where_speech_is_deemed_probably_absent))/(current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold-1) , 0);
    end
    
    %LARGE WINDOW:
    if flag_omlsa_use_large_local_indicator==1
        indices_where_speech_is_deemed_absolutely_absent2 = ...
            (current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2<=1 & ...
            smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2<smoothed_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold);
        indices_where_speech_is_deemed_probably_present2 = not(indices_where_speech_is_deemed_absolutely_absent2);
        indices_where_speech_is_deemed_probably_absent2 = ...
            (current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2>1 & ...
            current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2<current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold & ...
            smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2<smoothed_raw_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold);
        %put 1s in speech absence probability where i found speech to be absent with very high probability:
        if (~isempty(indices_where_speech_is_deemed_absolutely_absent2))
            q_apriori_speech_absence_probability2(indices_where_speech_is_deemed_absolutely_absent2) = 1;
        end
        %initialize 0 in all places where speech is not deemed absolutely absent:
        if (~isempty(indices_where_speech_is_deemed_probably_present2))
            q_apriori_speech_absence_probability2(indices_where_speech_is_deemed_probably_present2) = 0;
        end
        %use linear soft decision where speech is deemed only probably absent:
        if (~isempty(indices_where_speech_is_deemed_probably_absent2))
            q_apriori_speech_absence_probability2(indices_where_speech_is_deemed_probably_absent2) = max( (current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold-current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2(indices_where_speech_is_deemed_probably_absent2))/(current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold-1) , 0);
        end
        q_apriori_speech_absence_probability = (q_apriori_speech_absence_probability.*q_apriori_speech_absence_probability2).^(1/2);
        indices_where_speech_is_deemed_probably_present = indices_where_speech_is_deemed_probably_present.*indices_where_speech_is_deemed_probably_present2;
    end
    
end
%limit q_apriori_speech_apsence_probability_if_wanted
q_apriori_speech_absence_probability = max(min(q_apriori_speech_absence_probability, 1),0);


%Smooth apriori speech absence probability
if flag_smooth_apriori_speech_absence_probability==1 && number_of_bins_to_smooth_speech_presence_probability>1
    if flag_smooth_speech_presence_linear_logarithmic_or_raised_decay==1
        q_apriori_speech_absence_probability = conv_without_end_effects(q_apriori_speech_absence_probability,hanning(1,number_of_bins_to_smooth_speech_presence_probability));
    elseif flag_smooth_speech_presence_linear_logarithmic_or_raised_decay==2
        q_apriori_speech_absence_probability_log = log(q_apriori_speech_absence_probability);
        q_apriori_speech_absence_probability_log_averaged = conv_without_end_effects(q_apriori_speech_absence_probability_log,hanning(1,number_of_bins_to_smooth_speech_presence_probability));
        q_apriori_speech_absence_probability = exp(q_apriori_speech_absence_probability_log_averaged);
    else
        q_apriori_speech_absence_probability = smooth_vec_by_decay(q_apriori_speech_absence_probability,raised_decay_dB_per_bin,number_of_bins_to_smooth_speech_presence_probability);
    end
end

%Get final speech presence probability:
%p=1./(1+((q./(1-q)).*(1+eps_cap).*exp(-v)));
p_speech_presence_probability = zeros(current_frame_size,1);
flag_use_IMCRA_or_MY_method_for_speech_presence_probability = 2;

if flag_use_IMCRA_or_MY_method_for_speech_presence_probability==1
    %USE IMCRA METHOD:
if (~isempty(indices_where_speech_is_deemed_probably_present))
    temp1 = q_apriori_speech_absence_probability(indices_where_speech_is_deemed_probably_present)./(1-q_apriori_speech_absence_probability(indices_where_speech_is_deemed_probably_present));
    temp2 = 1 + current_apriori_SNR_smoothed_estimate(indices_where_speech_is_deemed_probably_present);
    temp3 = exp(-v_k(indices_where_speech_is_deemed_probably_present));
    p_speech_presence_probability(indices_where_speech_is_deemed_probably_present) = (1 + temp1.*temp2.*temp3).^-1;
end
elseif flag_use_IMCRA_or_MY_method_for_speech_presence_probability==2
%USE MY METHOD:
%Assign simpliying noise and clean signal variance variables:
Yk2_smoothed = raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current.*final_noise_ps_after_beta_correction_previous;
lambda_d = final_noise_ps_after_beta_correction_previous;
lambda_x = final_noise_ps_after_beta_correction_previous .* current_apriori_SNR_smoothed_estimate;

%Calculate conditional probabilities:
probability_density_for_Yk_assuming_speech_is_absent = (1/pi./lambda_d) .* exp(-Yk2_smoothed./lambda_d);
probability_density_for_Yk_assuming_speech_is_present = (1/pi./(lambda_d+lambda_x)) .* exp(-Yk2_smoothed./(lambda_d+lambda_x));

%Total probability for current noisy speech frame:
p_apriori_speech_presence_probability = 1-q_apriori_speech_absence_probability;
total_probability_for_current_noisy_speech_frame = p_apriori_speech_presence_probability.*probability_density_for_Yk_assuming_speech_is_present + ...
    + q_apriori_speech_absence_probability.*probability_density_for_Yk_assuming_speech_is_absent;
p_speech_presence_probability = (p_apriori_speech_presence_probability.*probability_density_for_Yk_assuming_speech_is_present) ./ total_probability_for_current_noisy_speech_frame;
end

p_speech_presence_probability = min(p_speech_presence_probability,p_speech_presence_probability_maximum);
p_speech_presence_probability = max(p_speech_presence_probability,0);

 
%Smooth speech presence probability if wanted:
if flag_smooth_speech_presence_probability==1 && number_of_bins_to_smooth_speech_presence_probability>1
    if flag_smooth_speech_presence_linear_logarithmic_or_raised_decay==1
        p_speech_presence_probability = conv_without_end_effects(p_speech_presence_probability,hanning(1,number_of_bins_to_smooth_speech_presence_probability));
    elseif flag_smooth_speech_presence_linear_logarithmic_or_raised_decay==2
        p_speech_presence_probability_log = log(p_speech_presence_probability);
        p_speech_presence_probability_log_averaged = conv_without_end_effects(p_speech_presence_probability_log,hanning(1,number_of_bins_to_smooth_speech_presence_probability));
        p_speech_presence_probability = exp(p_speech_presence_probability_log_averaged);
    else
        p_speech_presence_probability = smooth_vec_by_decay(p_speech_presence_probability,raised_decay_dB_per_bin,number_of_bins_to_smooth_speech_presence_probability);
    end
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%TRACK MINIMUM USING SUB-FRAMES (SORTING OBJECT IS TOO SLOW):
spectrum_minimum_buildup_counter = spectrum_minimum_buildup_counter+1;
if spectrum_minimum_buildup_counter==number_of_frames_to_buildup_spectrum_minimum
   
    %RAW SMALL WINDOW:
    stored_raw_window_averaged_ps_smoothed_min(:,u1) = raw_frequency_smoothed_ps_time_smoothed_min_sw;
    u1 = u1+1;
    if u1==number_of_spectral_minimum_buildup_blocks_to_remember+1; 
       u1 = 1;
    end
    raw_frequency_smoothed_ps_time_smoothed_min = min(stored_raw_window_averaged_ps_smoothed_min,[],2);
    raw_frequency_smoothed_ps_time_smoothed_min_sw = raw_frequency_smoothed_ps_time_smoothed;
    
    %RAW LARGE WINDOW:
    stored_raw_window_averaged_ps_smoothed_min2(:,u2) = raw_frequency_smoothed_ps_time_smoothed_min_sw2;
    u2 = u2+1;
    if u2==number_of_spectral_minimum_buildup_blocks_to_remember+1; 
       u2 = 1;
    end
    raw_frequency_smoothed_ps_time_smoothed_min2 = min(stored_raw_window_averaged_ps_smoothed_min2,[],2);
    raw_frequency_smoothed_ps_time_smoothed_min_sw2 = raw_frequency_smoothed_ps_time_smoothed2;
    
    %PRIMARILY NOISE SMALL WINDOW:
    stored_noise_window_averaged_ps_smoothed_min(:,u3) = raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw;
    u3 = u3+1;
    if u3==number_of_spectral_minimum_buildup_blocks_to_remember+1; 
       u3 = 1;
    end
    raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min = min(stored_noise_window_averaged_ps_smoothed_min,[],2);
    raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw = raw_frequency_smoothed_primarily_noise_ps_time_smoothed;
    
    %PRIMARILY NOISE LARGE WINDOW:
    stored_noise_window_averaged_ps_smoothed_min2(:,u4) = raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw2;
    u4 = u4+1;
    if u4==number_of_spectral_minimum_buildup_blocks_to_remember+1; 
       u4 = 1;
    end
    raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2 = min(stored_noise_window_averaged_ps_smoothed_min2,[],2);
    raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min_sw2 = raw_frequency_smoothed_primarily_noise_ps_time_smoothed2;
    
    %initialize spectrum minimum counter:
    spectrum_minimum_buildup_counter = 0;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALCULATE FINAL NOISE PS USING SPEECH PRESENCE PROBABILITY:

soft_decision_where_to_update_final_noise_ps = final_noise_speech_activity_probability_smoothing_factor ...
 + (1-final_noise_speech_activity_probability_smoothing_factor)*p_speech_presence_probability;


final_noise_ps_before_beta_correction_previous = soft_decision_where_to_update_final_noise_ps.*final_noise_ps_before_beta_correction_previous ...
                                                + (1-soft_decision_where_to_update_final_noise_ps).*current_frame_ps; 

final_noise_ps_after_beta_correction_previous = final_noise_correction_factor * final_noise_ps_before_beta_correction_previous;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculate minimum attenuation correction factor when considering tonals:
if flag_remove_interfering_tonals==1  
    final_noise_ps_after_beta_correction_previous(4:end-3) = ...
        min([final_noise_ps_after_beta_correction_previous(4:end-3),...
            final_noise_ps_after_beta_correction_previous(1:end-6),...
            final_noise_ps_after_beta_correction_previous(7:end)],[],2);
    less_smoothed_raw_ps = 0.8*less_smoothed_raw_ps+0.2*current_frame_ps; 
    minimum_attenuation_correction_factor = (final_noise_ps_after_beta_correction_previous./(less_smoothed_raw_ps+1e-10)).^0.5;
else
    minimum_attenuation_correction_factor = 1; %DON'T FORGET TO INCORPORATE THIS ABOVE
end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%UPDATE PARAMETERS FOR NEXT ROUND:
frame_counter = frame_counter + 1;
raw_ps_to_smoothed_noise_ps_aposteriori_SNR_previous = raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current;
spectral_magnitude_gain_function_previous = spectral_magnitude_gain_function_current;
smoothed_apriori_SNR_frequency_averaged_previous = current_apriori_SNR_smoothed_estimate;

end

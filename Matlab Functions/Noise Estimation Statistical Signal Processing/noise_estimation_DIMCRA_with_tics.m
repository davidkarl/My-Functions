function parameters = noise_estimation_DIMCRA_with_tics(current_frame_ps,parameters)
%         parameters = struct('n',2,'len',len_val,'noise_cap',ns_ps,'noise_tild',ns_ps,'gamma',ones(len_val,1),'Sf',Sf_val,...
%             'Smin',Sf_val,'S',Sf_val,'S_tild',Sf_val,'GH1',ones(len_val,1),'Smin_tild',Sf_val,'Smin_sw',Sf_val,'Smin_sw_tild',Sf_val,...
%             'stored_min',max(ns_ps)*ones(len_val,U_val),'stored_min_tild',max(ns_ps)*ones(len_val,U_val)','u1',1,'u2',1,'j',2,...
%             'alpha_d',0.85,'alpha_s',0.9,'U',8,'V',15,'Bmin',1.66,'gamma0',4.6,'gamma1',3,'psi0',1.67,'alpha',0.92,'beta',1.47,...
%             'b',b_val,'Sf_tild',Sf_tild_val);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GET PARAMETERS FROM PREVIOUS SESSION:
tic
%frame size:
current_frame_size = parameters.len;

%counters:
frame_counter = parameters.n;
spectrum_minimum_buildup_counter = parameters.j;
number_of_spectral_minimum_buildup_blocks_to_remember = parameters.U;
number_of_frames_to_buildup_spectrum_minimum = parameters.V;
total_number_of_frames_where_minimum_is_searched = number_of_spectral_minimum_buildup_blocks_to_remember*number_of_frames_to_buildup_spectrum_minimum;

u1 = parameters.u1;
u2 = parameters.u2;

%raw power spectrums:
%SMALL WINDOW:
raw_frequency_smoothed_ps_time_smoothed = parameters.S;
%LARGE WINDOW:
raw_frequency_smoothed_ps_time_smoothed2 = parameters.S_large_smooth;

%raw power spectrums containing primarily noise:
%SMALL WINDOW:
raw_frequency_smoothed_primarily_noise_ps_time_smoothed = parameters.S_tild;
%LARGE WINDOW:
raw_frequency_smoothed_primarily_noise_ps_time_smoothed2 = parameters.S_tild_large_smooth;

%Smoothing factors:
apriori_SNR_smoothing_factor = parameters.alpha;
raw_frequency_smoothed_ps_time_smoothing_factor = parameters.alpha_s;
final_noise_speech_activity_probability_smoothing_factor = parameters.alpha_d;

%PS to PS minimum ratio SNR thresholds:
current_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold = parameters.gamma0;
smoothed_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold = parameters.psi0;

%Speech presence probability threshold:
current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold = parameters.gamma1;

%Spectrum smoothing window:
small_local_frequency_window_smoother = parameters.small_local_frequency_window;
large_local_frequency_window_smoother = parameters.large_local_frequency_window;

%Get previous session spectrum and SNR parameters:
raw_ps_to_smoothed_noise_ps_aposteriori_SNR_previous = parameters.gamma;
final_noise_ps_after_beta_correction_previous = parameters.noise_ps;
final_noise_ps_before_beta_correction_previous = parameters.noise_tild;

%Previous (log-MMSE) gain function (GH1):
spectral_magnitude_gain_function_previous = parameters.GH1;

%Minimum to nominal ps correction factor:
minimum_ps_correction_factor = parameters.Bmin;

%Final noise correction factor (beta):
final_noise_correction_factor = parameters.beta;

%Max aposteriori SNR:
maximum_raw_aposteriori_SNR = parameters.maximum_raw_aposteriori_SNR;

%use median or min to decide raw frequency and time smoothed aposteriori SNR:
flag_use_median_or_min_in_initial_aposteriori_SNR_VAD = parameters.flag_us_median_or_min_in_initial_aposterioi_SNR_VAD; 
flag_use_median_or_min_in_mainly_noise_aposteriori_SNR = parameters.flag_use_median_or_min_in_final_mainly_noise_aposteriori_SNR;

%sorting objects:
running_raw_frequency_time_smoothed_ps_sorting_object = parameters.running_raw_ps_sorting_object;
running_mainly_noise_frequency_smoothed_sorting_object = parameters.running_mainly_noise_ps_sorting_object;
running_raw_frequency_time_smoothed_ps_sorting_object_large = parameters.running_raw_ps_large_smooth_sorting_object;
running_mainly_noise_frequency_smoothed_sorting_object_large = parameters.running_mainly_noise_large_smooth_ps_sorting_object;

%get gain method:
flag_gain_method = parameters.flag_gain_method;
bayesian_cost_function_p_power = parameters.bayesian_cost_function_p_power;

%Get previous smoothed apriori SNR sum over all frequencies:
smoothed_apriori_SNR_frequency_averaged_previous = parameters.smoothed_apriori_SNR_frequency_averaged_previous;
smoothed_apriori_SNR_constrained_peak = parameters.smoothed_apriori_SNR_constrained_peak;

%speech presence and absence probability maximums:
flag_use_omlsa_speech_absence_or_minima_controled = parameters.flag_use_omlsa_speech_absence_or_minima_controled;
flag_omlsa_use_small_local_indicator = parameters.flag_omlsa_use_small_local_indicator;
flag_omlsa_use_large_local_indicator = parameters.flag_omlsa_use_large_local_indicator;
flag_omlsa_use_global_indicator = parameters.flag_omlsa_use_global_indicator;
q_apriori_speech_absence_probability_maximum = parameters.q_apriori_speech_absence_probability_maximum;
p_speech_presence_probability_maximum = parameters.p_speech_presence_probability_maximum;

%get speech probability smoothing parameters:
flag_smooth_speech_presence_probability = parameters.flag_smooth_speech_presence_probability;
flag_smooth_apriori_speech_absence_probability = parameters.flag_smooth_apriori_speech_absence_probability;
number_of_bins_to_smooth_speech_presence_probability = parameters.number_of_bins_to_smooth_speech_presence_probability;
flag_smooth_speech_presence_linear_logarithmic_or_raised_decay = parameters.flag_smooth_speech_presence_linear_logarithmic_or_raised_decay;
raised_decay_dB_per_bin = parameters.raised_decay_dB_per_bin;

%stored specral minimas:
stored_raw_window_averaged_ps_smoothed_min = parameters.stored_min;
stored_noise_window_averaged_ps_smoothed_min = parameters.stored_min_tild;

toc
disp('*********************************************************************');
disp('get variables from last round');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UPDATE AND CALCULATE THINGS INVOLVING RAW (SIGNAL ONLY) POWER SPECTRUM:
%Calculate current raw ps to smoothed noise ps aposteriori SNR (gamma): 
tic
raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current = current_frame_ps ./ final_noise_ps_after_beta_correction_previous;
toc
disp('calculate raw ps to smoothed noise minB aposteriori SNR current');

%limit aposteriori SNR:
tic
raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current = min(raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current,maximum_raw_aposteriori_SNR);
toc
disp('limit raw ps to smoothed noise minB ps aposteriori SNR current');

%Calculate apriori SNR estimate (epsilon):
tic
previous_smoothed_apriori_SNR_estimate = spectral_magnitude_gain_function_previous.^2 .* raw_ps_to_smoothed_noise_ps_aposteriori_SNR_previous;
current_raw_apriori_SNR_estimate_using_aposteriori_SNR_estimate = max(raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current-1,0); 
current_apriori_SNR_smoothed_estimate = apriori_SNR_smoothing_factor * previous_smoothed_apriori_SNR_estimate ...
    + (1-apriori_SNR_smoothing_factor) * current_raw_apriori_SNR_estimate_using_aposteriori_SNR_estimate;
toc
disp('estimation smoothed apriori SNR using decision directed approach');

%Compute current v_k:
tic
v_k = (current_apriori_SNR_smoothed_estimate./(1+current_apriori_SNR_smoothed_estimate)) .* raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current;
toc
disp('calculate v_k');


%Compute Spectral Gain According to Desired Algorithm:
tic
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
toc
disp('calculate gain function using gain method');

%Calculate window-averaged RAW power spectrum (SIGNAL+NOISE):
%SMALL WINDOW:
tic
[raw_frequency_smoothed_ps] = conv_without_end_effects(current_frame_ps,small_local_frequency_window_smoother);
%LARGE WINDOW: 
[raw_frequency_smoothed_ps2] = conv_without_end_effects(current_frame_ps,large_local_frequency_window_smoother);
toc
disp('conv without end effects to frequency smooth small and large window');

%Smooth raw frequency smoothed ps in time domain:
%Small window:
tic
raw_frequency_smoothed_ps_time_smoothed = raw_frequency_smoothed_ps_time_smoothing_factor*raw_frequency_smoothed_ps_time_smoothed ... 
                                + (1-raw_frequency_smoothed_ps_time_smoothing_factor)*raw_frequency_smoothed_ps;
%Large window:
raw_frequency_smoothed_ps_time_smoothed2 = raw_frequency_smoothed_ps_time_smoothing_factor*raw_frequency_smoothed_ps_time_smoothed2 ... 
                                + (1-raw_frequency_smoothed_ps_time_smoothing_factor)*raw_frequency_smoothed_ps2;
toc
disp('smooth in time domain raw frequency ps');
                            
                            
%Calculate ps to smoothed Pmin aposteriori SNR (gamma_min) and smooth ps to smooth ps minimum (psi_min):
%Use sorting object to track min and median of raw frequency and time smoothed ps:
%SMALL WINDOW:
tic
[time_smoothed_raw_frequency_and_time_smoothed_median,raw_frequency_smoothed_ps_time_smoothed_min,running_max_vec] = ...
    running_raw_frequency_time_smoothed_ps_sorting_object.update_median(raw_frequency_smoothed_ps_time_smoothed);
toc
disp('update median of small window frequency smoothed ps');

tic
if flag_use_median_or_min_in_initial_aposteriori_SNR_VAD==0
    current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min = current_frame_ps ./ time_smoothed_raw_frequency_and_time_smoothed_median;
    smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min = raw_frequency_smoothed_ps_time_smoothed ./ time_smoothed_raw_frequency_and_time_smoothed_median;
else
    current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min = current_frame_ps ./ (minimum_ps_correction_factor*raw_frequency_smoothed_ps_time_smoothed_min);
    smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min = raw_frequency_smoothed_ps_time_smoothed ./ (minimum_ps_correction_factor*raw_frequency_smoothed_ps_time_smoothed_min);
end
toc
disp('calculate aposteriori SNRs using minB small window');

%LARGE WINDOW:
tic
[time_smoothed_raw_frequency_and_time_smoothed_median2,raw_frequency_smoothed_ps_time_smoothed_min2,running_max_vec] = ...
    running_raw_frequency_time_smoothed_ps_sorting_object.update_median(raw_frequency_smoothed_ps_time_smoothed2);
toc 
disp('update median of large window frequency smoothed ps');

tic
if flag_use_median_or_min_in_initial_aposteriori_SNR_VAD==0
    current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min2 = current_frame_ps ./ time_smoothed_raw_frequency_and_time_smoothed_median2;
    smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min2 = raw_frequency_smoothed_ps_time_smoothed2 ./ time_smoothed_raw_frequency_and_time_smoothed_median2;
else
    current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min2 = current_frame_ps ./ (minimum_ps_correction_factor*raw_frequency_smoothed_ps_time_smoothed_min2);
    smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min2 = raw_frequency_smoothed_ps_time_smoothed2 ./ (minimum_ps_correction_factor*raw_frequency_smoothed_ps_time_smoothed_min2);
end
toc
disp('calculate aposteriori SNRs using minB large window');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UPDATE AND CALCULATE THINGS INVOLVING PRIMARILY NOISE POWER SPECTRUM:
%%%% SMALL WINDOW:
tic
%Build logical mask which has ones where speech is deemed absent (and we might need to update ps around it):
logical_mask_where_speech_is_absent = zeros(current_frame_size,1);
logical_mask_where_speech_is_absent(...
    current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min < current_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold ... 
    & smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min < smoothed_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold) = 1;  
toc
disp('get logical mask small frequency smoothed');
 
%Update Sf_tilde to update S_tilde in frequencies where in their vicinity were found to contain only noise:
tic
[raw_frequency_smoothed_primarily_noise_ps] = average_over_indicators_and_smoothing_window(current_frame_ps,logical_mask_where_speech_is_absent,small_local_frequency_window_smoother,raw_frequency_smoothed_primarily_noise_ps_time_smoothed);
raw_frequency_smoothed_primarily_noise_ps_time_smoothed = raw_frequency_smoothed_ps_time_smoothing_factor*raw_frequency_smoothed_primarily_noise_ps_time_smoothed ...
                                                + (1-raw_frequency_smoothed_ps_time_smoothing_factor)*raw_frequency_smoothed_primarily_noise_ps;
toc
disp('smooth over indicators small frequency window');

%Calculate noise ps to smoothed P_noise_min aposteriori SNR (gamma_min_tilde) 
%and smooth noise ps to smooth noise ps minimum (psi_min_tilde):
%Use sorting object to track min and median of raw frequency and time smoothed mainly noise ps:
tic
[raw_frequency_smoothed_primarily_noise_ps_time_smoothed_median,raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min,running_max_vec] = ...
    running_mainly_noise_frequency_smoothed_sorting_object.update_median(raw_frequency_smoothed_primarily_noise_ps_time_smoothed);
if flag_use_median_or_min_in_mainly_noise_aposteriori_SNR==0
    current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR = current_frame_ps ./ raw_frequency_smoothed_primarily_noise_ps_time_smoothed_median;
    smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR = raw_frequency_smoothed_primarily_noise_ps_time_smoothed ./ raw_frequency_smoothed_primarily_noise_ps_time_smoothed_median;
else
    current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR = current_frame_ps ./ (minimum_ps_correction_factor*raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min);
    smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR = raw_frequency_smoothed_primarily_noise_ps_time_smoothed ./ (minimum_ps_correction_factor*raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min);
end
toc
disp('get small window primarily noise median and aposterioris');


%%%% LARGE WINDOW:
%Build logical mask which has ones where speech is deemed absent (and we might need to update ps around it):
tic
logical_mask_where_speech_is_absent2 = zeros(current_frame_size,1);
logical_mask_where_speech_is_absent2(...
    current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min2 < current_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold ... 
    & smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min2 < smoothed_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold) = 1;  
toc
disp('get logical mask large window');
 
%Update Sf_tilde to update S_tilde in frequencies where in their vicinity were found to contain only noise:
tic
[raw_frequency_smoothed_primarily_noise_ps2] = average_over_indicators_and_smoothing_window(current_frame_ps,logical_mask_where_speech_is_absent2,large_local_frequency_window_smoother,raw_frequency_smoothed_primarily_noise_ps_time_smoothed);
raw_frequency_smoothed_primarily_noise_ps_time_smoothed2 = raw_frequency_smoothed_ps_time_smoothing_factor*raw_frequency_smoothed_primarily_noise_ps_time_smoothed2 ...
                                                + (1-raw_frequency_smoothed_ps_time_smoothing_factor)*raw_frequency_smoothed_primarily_noise_ps2;
toc
disp('smooth over indicators large window');
                                            
%Calculate noise ps to smoothed P_noise_min aposteriori SNR (gamma_min_tilde) 
%and smooth noise ps to smooth noise ps minimum (psi_min_tilde):
%Use sorting object to track min and median of raw frequency and time smoothed mainly noise ps:
tic
[raw_frequency_smoothed_primarily_noise_ps_time_smoothed_median2,raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2,running_max_vec] = ...
    running_mainly_noise_frequency_smoothed_sorting_object.update_median(raw_frequency_smoothed_primarily_noise_ps_time_smoothed2);
toc
disp('get large window primarily noise median and aposterioris');

tic
if flag_use_median_or_min_in_mainly_noise_aposteriori_SNR==0
    current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2 = current_frame_ps ./ raw_frequency_smoothed_primarily_noise_ps_time_smoothed_median2;
    smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2 = raw_frequency_smoothed_primarily_noise_ps_time_smoothed2 ./ raw_frequency_smoothed_primarily_noise_ps_time_smoothed_median2;
else
    current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2 = current_frame_ps ./ (minimum_ps_correction_factor*raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2);
    smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2 = raw_frequency_smoothed_primarily_noise_ps_time_smoothed2 ./ (minimum_ps_correction_factor*raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min2);
end
toc
disp('calculate logical mask and aposteriori SNR where there is mainly noise according to large frequency smoothed window');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v%%%%%%%%%%%%%%%%%%%%
%CALCULATE APPROXIMATE A-PRIORI SPEECH ABSENCE PROBABILITY AND CONDITIONAL SPEECH PRESENCE PROBABILITY:
tic
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
    smoothed_apriori_SNR_small_local_average = smooth_vector_over_N_neighbors(current_apriori_SNR_smoothed_estimate,1);
    P_contribution_by_small_local_sum_of_apriori_SNR = log10(smoothed_apriori_SNR_small_local_average/smoothed_apriori_SNR_minimum) / log10(smoothed_apriori_SNR_maximum/smoothed_apriori_SNR_minimum);
    indices_where_small_local_speech_presence_probability_is_small = (smoothed_apriori_SNR_small_local_average < smoothed_apriori_SNR_minimum);
    indices_where_small_local_speech_presence_probability_is_medium = ((smoothed_apriori_SNR_small_local_average > smoothed_apriori_SNR_minimum) & (smoothed_apriori_SNR_small_local_average < smoothed_apriori_SNR_maximum));
    indices_where_small_local_speech_presence_probability_is_large = (smoothed_apriori_SNR_small_local_average > smoothed_apriori_SNR_maximum);
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_small) = 0;
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_medium) = P_contribution_by_small_local_sum_of_apriori_SNR(indices_where_small_local_speech_presence_probability_is_medium);
    P_speech_presence_probability_in_small_local_frequency_range( indices_where_small_local_speech_presence_probability_is_large) = 1;
    
    %Estimate speech presence probability inside a large local area on the frequency vec:
    xi_k_smoothed_large_local_average = smooth_vector_over_N_neighbors(current_apriori_SNR_smoothed_estimate,15);
    P_contribution_by_large_local_sum_of_apriori_SNR = log10(xi_k_smoothed_large_local_average/smoothed_apriori_SNR_minimum) / log10(smoothed_apriori_SNR_maximum/smoothed_apriori_SNR_minimum);
    indices_where_large_local_speech_presence_probability_is_small = (xi_k_smoothed_large_local_average < smoothed_apriori_SNR_minimum);
    indices_where_large_local_speech_presence_probability_is_medium = ((xi_k_smoothed_large_local_average > smoothed_apriori_SNR_minimum) & (xi_k_smoothed_large_local_average < smoothed_apriori_SNR_maximum));
    indices_where_large_local_speech_presence_probability_is_large = (xi_k_smoothed_large_local_average > smoothed_apriori_SNR_maximum);
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_small) = 0;
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_medium) = P_contribution_by_large_local_sum_of_apriori_SNR(indices_where_large_local_speech_presence_probability_is_medium);
    P_speech_presence_probability_in_large_local_frequency_range( indices_where_large_local_speech_presence_probability_is_large) = 1;
    
    %Estimate speech presence probability over all frequency vec:
    smoothed_apriori_SNR_averaged_over_all_frequencies_current = mean(current_apriori_SNR_smoothed_estimate);
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
                P_contribution_from_apriori_SNR_global_sum = 0;
            elseif smoothed_apriori_SNR_averaged_over_all_frequencies_current >= smoothed_apriori_SNR_constrained_peak*smoothed_apriori_SNR_maximum
                P_contribution_from_apriori_SNR_global_sum = 1;
            else
                P_contribution_from_apriori_SNR_global_sum = log10(smoothed_apriori_SNR_averaged_over_all_frequencies_current/smoothed_apriori_SNR_constrained_peak/smoothed_apriori_SNR_minimum) / log10(smoothed_apriori_SNR_maximum/smoothed_apriori_SNR_minimum);
            end
        end
    else
        P_contribution_from_apriori_SNR_global_sum = 0;
    end
    
    %Get normalizing power:
    normalizing_probability_power_for_geometric_mean = 0;
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
    P_apriori_speech_presence_probability = P_apriori_speech_presence_probability.^(1/normalizing_probability_power_for_geometric_mean);
    

    %Estimate final speech presence probability probability using Cohen's method:
    q_apriori_speech_absence_probability = 1 - P_apriori_speech_presence_probability;

    %Assign indices where speech probability is to updated to all:
    indices_where_speech_is_deemed_maybe_present = (1:length(current_frame_ps))';
else
    
    %Use Original IMCRA method (direct estimation of q apriori speech ABSENCE):
    
    %SMALL WINDOW:
    indices_where_speech_is_deemed_absolutely_absent = find(current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR<=1 & ...
        smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR < smoothed_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold);
    indices_where_speech_is_deemed_maybe_present = setdiff([1:current_frame_size],indices_where_speech_is_deemed_absolutely_absent);
    indices_where_speech_is_deemed_probably_absent = find(current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR>1 & current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR<current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold & smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR<smoothed_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold);
    %put 1s in speech absence probability where i found speech to be absent with very high probability:
    if (~isempty(indices_where_speech_is_deemed_absolutely_absent))
        q_apriori_speech_absence_probability(indices_where_speech_is_deemed_absolutely_absent) = 1;
    end
    %initialize 0 in all places where speech is not deemed absolutely absent:
    if (~isempty(indices_where_speech_is_deemed_maybe_present))
        q_apriori_speech_absence_probability(indices_where_speech_is_deemed_maybe_present) = 0;
    end
    %use linear soft decision where speech is deemed only probably absent:
    if (~isempty(indices_where_speech_is_deemed_probably_absent))
        q_apriori_speech_absence_probability(indices_where_speech_is_deemed_probably_absent) = max( (current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold-current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR(indices_where_speech_is_deemed_probably_absent))/(current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold-1) , 0);
    end
    
    %LARGE WINDOW:
    if flag_omlsa_use_large_local_indicator==1
        indices_where_speech_is_deemed_absolutely_absent2 = find(current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2<=1 & ...
            smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2 < smoothed_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold);
        indices_where_speech_is_deemed_maybe_present2 = setdiff([1:current_frame_size],indices_where_speech_is_deemed_absolutely_absent2);
        indices_where_speech_is_deemed_probably_absent2 = find(current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2>1 & current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2<current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold & smoothed_ps_to_smoothed_noise_ps_minB_aposteriori_SNR2<smoothed_ps_to_smoothed_ps_minB_aposteriori_SNR_threshold);
        %put 1s in speech absence probability where i found speech to be absent with very high probability:
        if (~isempty(indices_where_speech_is_deemed_absolutely_absent2))
            q_apriori_speech_absence_probability2(indices_where_speech_is_deemed_absolutely_absent2) = 1;
        end
        %initialize 0 in all places where speech is not deemed absolutely absent:
        if (~isempty(indices_where_speech_is_deemed_maybe_present2))
            q_apriori_speech_absence_probability2(indices_where_speech_is_deemed_maybe_present2) = 0;
        end
        %use linear soft decision where speech is deemed only probably absent:
        if (~isempty(indices_where_speech_is_deemed_probably_absent2))
            q_apriori_speech_absence_probability2(indices_where_speech_is_deemed_probably_absent2) = max( (current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold-current_raw_ps_to_smoothed_noise_ps_minB_aposteriori_SNR(indices_where_speech_is_deemed_probably_absent2))/(current_raw_ps_to_smoothed_noise_ps_minB_upper_threshold-1) , 0);
        end
        q_apriori_speech_absence_probability = (q_apriori_speech_absence_probability.*q_apriori_speech_absence_probability2).^(1/2);
    end
    
end
%limit q_apriori_speech_apsence_probability_if_wanted
q_apriori_speech_absence_probability = max(min(q_apriori_speech_absence_probability, q_apriori_speech_absence_probability_maximum),0);
toc
disp('calculate q apriori speech absence probability');

%Smooth apriori speech absence probability
tic
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
toc
disp('smoothed q apriori speech absence probability if wanted');

%Get final speech presence probability:
%p=1./(1+((q./(1-q)).*(1+eps_cap).*exp(-v)));
tic
p_speech_presence_probability = zeros(current_frame_size,1);
if (~isempty(indices_where_speech_is_deemed_maybe_present))
    temp1 = q_apriori_speech_absence_probability(indices_where_speech_is_deemed_maybe_present)./(1-q_apriori_speech_absence_probability(indices_where_speech_is_deemed_maybe_present));
    temp2 = 1 + current_apriori_SNR_smoothed_estimate(indices_where_speech_is_deemed_maybe_present);
    temp3 = exp(-v_k(indices_where_speech_is_deemed_maybe_present));
    p_speech_presence_probability(indices_where_speech_is_deemed_maybe_present) = (1 + temp1.*temp2.*temp3).^-1;
end
p_speech_presence_probability = min(p_speech_presence_probability,p_speech_presence_probability_maximum);
p_speech_presence_probability = max(p_speech_presence_probability,0);
toc
disp('calculate final p speech presence probability');

%Smooth speech presence probability if wanted:
tic
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
toc
disp('smooth p speech presence probability in frequency');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALCULATE FINAL NOISE PS USING SPEECH PRESENCE PROBABILITY:
tic
soft_decision_where_to_update_final_noise_ps = final_noise_speech_activity_probability_smoothing_factor ...
 + (1-final_noise_speech_activity_probability_smoothing_factor)*p_speech_presence_probability;


final_noise_ps_before_beta_correction_current = soft_decision_where_to_update_final_noise_ps.*final_noise_ps_before_beta_correction_previous ...
                                                + (1-soft_decision_where_to_update_final_noise_ps).*current_frame_ps; 

             

final_noise_ps_current = final_noise_correction_factor * final_noise_ps_before_beta_correction_current;
toc
disp('calculate final noise ps using speech presence probability and B correction');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%UPDATE PARAMETERS FOR NEXT ROUND:
tic
parameters.n = frame_counter+1;
parameters.gamma = raw_ps_to_smoothed_noise_minB_ps_aposteriori_SNR_current;
parameters.GH1 = spectral_magnitude_gain_function_current;
parameters.Sf = raw_frequency_smoothed_ps;
parameters.Smin = raw_frequency_smoothed_ps_time_smoothed_min;
parameters.S = raw_frequency_smoothed_ps_time_smoothed;
parameters.S_tild = raw_frequency_smoothed_primarily_noise_ps_time_smoothed;
parameters.S_large_smooth = raw_frequency_smoothed_ps_time_smoothed2;
parameters.S_tild_large_smooth = raw_frequency_smoothed_primarily_noise_ps_time_smoothed2;
parameters.Smin_tild = raw_frequency_smoothed_primarily_noise_ps_time_smoothed_min;
parameters.stored_min = stored_raw_window_averaged_ps_smoothed_min;
parameters.stored_min_tild = stored_noise_window_averaged_ps_smoothed_min;
parameters.u1 = u1;
parameters.u2 = u2;
parameters.j = spectrum_minimum_buildup_counter;
parameters.noise_tild=final_noise_ps_before_beta_correction_current;
parameters.noise_ps = final_noise_ps_current;
parameters.speech_presence_probability = p_speech_presence_probability;
parameters.smoothed_apriori_SNR_frequency_averaged_previous = current_apriori_SNR_smoothed_estimate;
parameters.smoothed_apriori_SNR_constrained_peak = smoothed_apriori_SNR_constrained_peak;
parameters.flag_include_global_estimate_in_speech_absence_or_not = flag_use_omlsa_speech_absence_or_minima_controled;
parameters.q_apriori_speech_absence_probability_maximum = q_apriori_speech_absence_probability_maximum;
parameters.p_speech_presence_probability_maximum = p_speech_presence_probability_maximum;
parameters.flag_gain_method = flag_gain_method;
parameters.bayesian_cost_function_p_power = bayesian_cost_function_p_power;
parameters.running_raw_ps_sorting_object = running_raw_frequency_time_smoothed_ps_sorting_object;
parameters.running_mainly_noise_ps_sorting_object = running_mainly_noise_frequency_smoothed_sorting_object;
parameters.running_raw_ps_large_smooth_sorting_object = running_raw_frequency_time_smoothed_ps_sorting_object_large;
parameters.running_mainly_large_smooth_noise_ps_sorting_object = running_mainly_noise_frequency_smoothed_sorting_object_large;
parameters.maximum_raw_aposteriori_SNR = maximum_raw_aposteriori_SNR;
toc
disp('transfer variables for next round');
disp('*********************************************************************');






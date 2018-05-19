function parameters = noise_estimation_MCRA2(current_frame_power_spectrum,parameters)

% parameters = struct('n',2,'len',len_val,'ad',0.95,'as',0.8,'ap',0.2,'beta',0.8,'beta1',0.98,'gamma',0.998,'alpha',0.7,...
%             'pk',zeros(len_val,1),'noise_ps',ns_ps,'pxk_old',ns_ps,'pxk',ns_ps,'pnk_old',ns_ps,'pnk',ns_ps);

%Initialize parameters:
frame_counter = parameters.n;
current_frame_size = parameters.len;
noise_ps_smoothing_factor_smoothing_factor = parameters.ad;
signal_ps_raw_smoothing_factor = parameters.as;
noise_ps_smoothing_factor_update_mask_smoothing_factor = parameters.ap;
Pmin_update_beta = parameters.beta;
Pmin_estimate_smoothing_factor = parameters.gamma;
raw_ps_smoothing_factor = parameters.alpha;
noise_ps_smoothing_factor_update_mask_smoothed = parameters.pk;
ps_to_mps_ratio_threshold = parameters.delta;
noise_ps_final_smoothed = parameters.noise_ps;
raw_ps_estimate_current_smoothed = parameters.pxk;
Pmin_smoothed = parameters.pnk;
raw_ps_estimate_old = parameters.pxk_old;
Pmin_estimate_smoothed_old = parameters.pnk_old;


%Calculate raw power spectrum smoothed estimate:
raw_ps_estimate_current_smoothed = raw_ps_smoothing_factor*raw_ps_estimate_old ...
                                     + (1-raw_ps_smoothing_factor)*(current_frame_power_spectrum);

%Perform minimal tracking (Pmin) by finding where the current ps is lower then the
%minimum so far and assigning it to Pmin, and where its bigger perform a smoothing:
indices_where_raw_ps_current_is_larger_then_Pmin = find(Pmin_estimate_smoothed_old<raw_ps_estimate_current_smoothed);
indices_where_raw_ps_current_is_smaller_then_Pmin = setdiff(1:length(raw_ps_estimate_current_smoothed),indices_where_raw_ps_current_is_larger_then_Pmin);
Pmin_derivative_update_term = (raw_ps_estimate_current_smoothed - Pmin_update_beta*raw_ps_estimate_old)/(1-Pmin_update_beta);
Pmin_smoothed(indices_where_raw_ps_current_is_smaller_then_Pmin) = raw_ps_estimate_current_smoothed(indices_where_raw_ps_current_is_smaller_then_Pmin);
Pmin_smoothed(indices_where_raw_ps_current_is_larger_then_Pmin) = Pmin_estimate_smoothing_factor*Pmin_estimate_smoothed_old(indices_where_raw_ps_current_is_larger_then_Pmin) ...
    + (1-Pmin_estimate_smoothing_factor)*Pmin_derivative_update_term(indices_where_raw_ps_current_is_larger_then_Pmin);

%Assign current raw signal and noise power spectrum into old estimate variables:
raw_ps_estimate_old = raw_ps_estimate_current_smoothed;
Pmin_estimate_smoothed_old = Pmin_smoothed;

%Calculate SNR estimate vec and use threshold to create a logical mask which updates noise ps smoothing factor:
weird_SNR_estimate_smoothed_vec = raw_ps_estimate_current_smoothed ./ Pmin_smoothed;

%Calculate logical update mask for noise ps smoothing factor:
weird_SNR_ratio_above_threshold_logical_update_mask = zeros(current_frame_size,1);
weird_SNR_ratio_above_threshold_indices = find(weird_SNR_estimate_smoothed_vec > ps_to_mps_ratio_threshold);
weird_SNR_ratio_above_threshold_logical_update_mask(weird_SNR_ratio_above_threshold_indices) = 1;

%Calculate noise ps smoothing factor's smoothing factor using the logical mask from above:
noise_ps_smoothing_factor_update_mask_smoothed = noise_ps_smoothing_factor_update_mask_smoothing_factor*noise_ps_smoothing_factor_update_mask_smoothed ...
    + (1-noise_ps_smoothing_factor_update_mask_smoothing_factor)*weird_SNR_ratio_above_threshold_logical_update_mask;

%Calculate the noise ps' smoothing factor which is ideally only non-zero where there's only silence:
noise_ps_final_smoothing_factor = noise_ps_smoothing_factor_smoothing_factor ...
    + (1-noise_ps_smoothing_factor_smoothing_factor)*noise_ps_smoothing_factor_update_mask_smoothed;

%Calculate the final noise ps:
noise_ps_final_smoothed = noise_ps_final_smoothing_factor.*noise_ps_final_smoothed ...
    + (1-noise_ps_final_smoothing_factor).*raw_ps_estimate_current_smoothed;

%Assign final parameters for next estimate:
parameters.n = frame_counter+1;
parameters.pk = noise_ps_smoothing_factor_update_mask_smoothed;
parameters.noise_ps = noise_ps_final_smoothed;
parameters.pnk = Pmin_smoothed;
parameters.pnk_old = Pmin_estimate_smoothed_old;
parameters.pxk = raw_ps_estimate_current_smoothed;
parameters.pxk_old = raw_ps_estimate_old;

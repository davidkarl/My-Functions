function parameters = noise_estimation_MCRA(current_frame_power_spectrum,parameters)
%ps = power spectrum
%mps = minimum power spectrum

%Initialize parameters:
signal_ps_raw_smoothing_factor = parameters.as;
noise_ps_smoothing_factor_smoothing_factor = parameters.ad;
noise_ps_smoothing_factor_update_mask_smoothing_factor = parameters.ap;
noise_ps_smoothing_factor_update_mask_smoothed = parameters.pk;
ps_to_mps_ratio_threshold = parameters.delta;
number_of_frames_to_refresh_Ptmp = parameters.L;
frame_counter = parameters.n;
current_frame_size = parameters.len;
noise_ps_final_smoothed = parameters.noise_ps;
signal_ps_raw_smoothed = parameters.P;
Pmin = parameters.Pmin;
Ptmp = parameters.Ptmp;

%Calculate smoothed raw power spectrum:
signal_ps_raw_smoothed = signal_ps_raw_smoothing_factor*signal_ps_raw_smoothed + ...
    (1-signal_ps_raw_smoothing_factor)*current_frame_power_spectrum;

%Update Pmin & Ptmp and refresh Ptmp after certain number of frames:
if rem(frame_counter,number_of_frames_to_refresh_Ptmp)==0
    Pmin = min(Ptmp,signal_ps_raw_smoothed);
    Ptmp = signal_ps_raw_smoothed;
else
    Pmin = min(Pmin,signal_ps_raw_smoothed);
    Ptmp = min(Ptmp,signal_ps_raw_smoothed);
end

%Calculate vector of ratios between raw smoothed power spectrum values and minimum values up to that point:
smoothed_ps_to_mps_ratios = signal_ps_raw_smoothed ./ Pmin; 

%Calculate ps_to_mps logical update mask to update final noise ps smoothing factor:
ps_to_mps_ratio_above_threshold_logical_update_mask = zeros(current_frame_size,1);
ps_to_mps_ratio_above_threshold_indices = find(smoothed_ps_to_mps_ratios > ps_to_mps_ratio_threshold);
ps_to_mps_ratio_above_threshold_logical_update_mask(ps_to_mps_ratio_above_threshold_indices) = 1;

%Calculate final noise ps smoothing factor's smoothing logical update mask:
noise_ps_smoothing_factor_update_mask_smoothed = noise_ps_smoothing_factor_update_mask_smoothing_factor*noise_ps_smoothing_factor_update_mask_smoothed ...
                                                 + (1-noise_ps_smoothing_factor_update_mask_smoothing_factor)*ps_to_mps_ratio_above_threshold_logical_update_mask;

%Smooth noise power spectrum smoothing factor:
noise_ps_smoothing_factor = noise_ps_smoothing_factor_smoothing_factor ...
      + (1-noise_ps_smoothing_factor_smoothing_factor)*noise_ps_smoothing_factor_update_mask_smoothed;

%Estimate smoothed noise power spectrum:
noise_ps_final_smoothed = noise_ps_smoothing_factor.*noise_ps_final_smoothed ...
                             + (1-noise_ps_smoothing_factor).*current_frame_power_spectrum;

%Assign final parameters for next noise estimation:
parameters.pk = noise_ps_smoothing_factor_update_mask_smoothed;
parameters.n = frame_counter+1;
parameters.noise_ps = noise_ps_final_smoothed;
parameters.P = signal_ps_raw_smoothed;
parameters.Pmin = Pmin;
parameters.Ptmp = Ptmp;



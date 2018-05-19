function [noise_estimation_parameters,noise_gate_threshold_vec,flag_initialize,flag_restart_noise_estimation_parameters_according_to_GUI] = ...
    audio_estimate_noise_profile_parameters_united(noisy_signal_power_spectrum,method,noise_estimation_parameters,Fs,flag_initialize,coefficients_correction_factor)

if flag_initialize==1
    noise_estimation_parameters = initialise_noise_estimation_parameters(noisy_signal_power_spectrum,Fs,method,coefficients_correction_factor);
    flag_restart_noise_estimation_parameters_according_to_GUI = 1;
else    
    switch lower(method)
        case 'martin'
            noise_estimation_parameters = noise_estimation_martin(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'mcra'
            noise_estimation_parameters = noise_estimation_MCRA(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'imcra'
            noise_estimation_parameters = noise_estimation_IMCRA_fast_no_tics(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'dimcra'
            noise_estimation_parameters = noise_estimation_DIMCRA(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'doblinger'
            noise_estimation_parameters = noise_estimation_doblinger(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'hirsch'
            noise_estimation_parameters = noise_estimation_hirsch(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'mcra2'
            noise_estimation_parameters = noise_estimation_MCRA2(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'conn_freq'
            noise_estimation_parameters = noise_estimation_connected_time_frequency_regions(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'snr_dependent'
            noise_estimation_parameters = noise_estimation_SNR_dependent(noisy_signal_power_spectrum,noise_estimation_parameters);
    end
end
flag_initialize = 0;
noise_gate_threshold_vec = noise_estimation_parameters.noise_ps;
return;
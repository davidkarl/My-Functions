function [noise_estimation_parameters] = audio_estimate_noise_profile_parameters(noisy_signal_power_spectrum,method,noise_estimation_parameters)

    switch lower(method)
        case 'martin'
            noise_estimation_parameters = noise_estimation_martin(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'mcra'
            noise_estimation_parameters = noise_estimation_MCRA(noisy_signal_power_spectrum,noise_estimation_parameters);
        case 'imcra'
            noise_estimation_parameters = noise_estimation_IMCRA_fast_no_tics(noisy_signal_power_spectrum,noise_estimation_parameters);
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

return;
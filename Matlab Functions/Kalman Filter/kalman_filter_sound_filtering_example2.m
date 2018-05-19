function [y_signal_estimate,residual_error_over_time] = kalman_filter_sound_filtering_example2(input_signal)
% Filter segment once
% Takes a segment x of the noisy input signal, estimates parameters
% of its AR model and the noise covariances and applies Kalman
% filter to obtain the filtered signal y and innovation e
% Usage: [y e]=fltseg(x) ;

AR_model_order = 4; % predetermined order of the system
input_signal_length = length(input_signal);

% Estimate ARX parameters through autocorrelation
AR_parameters = get_AR_parameters_repetitive_with_weights(input_signal,AR_model_order);
AR_parameters = stabilize_AR_parameters_by_moving_them_inside_the_unit_circle(AR_parameters);
A_state_transition_matrix = AR_parameters_to_state_transfer_matrix(AR_parameters); % system matrix

% Estimate the noise covariances
number_of_auto_correlation_coefficients = min(input_signal_length,200) ; % number of coefficients

auto_correlation_sequence = ...
    get_auto_correlation_using_power_spectrum_and_ifft(input_signal,number_of_auto_correlation_coefficients) ; % calculate coefficients

[Q_process_noise , R_measurement_noise] = estimate_process_noise_from_auto_correlation_and_AR_parameters(...
                                                auto_correlation_sequence,AR_parameters) ; % process noise, measurement noise

% Calculate Kalman gain
kalman_gain_steady_state = get_steady_state_optimal_kalman_gain(...
                A_state_transition_matrix , Q_process_noise/R_measurement_noise) ;

% Filter
[y_signal_estimate , residual_error_over_time] = kalman_filter_steady_state_kalman_gain_filtering(...
                    input_signal,A_state_transition_matrix,kalman_gain_steady_state) ;






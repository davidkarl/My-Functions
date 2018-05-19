function [y_signal_estimate, total_error] = get_clean_signal_kalman_filter_with_signal_AR_and_noise_white(...
                                        input_signal,AR_parameters_signal,g_signal,R_noise_covariance)
% This function takes the corrupted signal z=signal+noise,
% AR parameters of signal (as,bs) and noise covariance.
% Noise is assumed to be white, Gaussian, with covariance r.
% Estimate y of the signal using bidirectional nonstationary Kalman filter.
% te is the total forward prediction error
% Usage: [y,te]=arkalm(z,as,bs,r) ;


[state_transition_matrix] = AR_parameters_to_state_transfer_matrix(AR_parameters);
AR_model_order = length(AR_parameters_signal);
input_signal_length = length(input_signal);
H_measurement_matrix = [zeros(1,AR_model_order-1) , 1];
Q_process_noise = zeros(AR_model_order); 
Q_process_noise(AR_model_order,AR_model_order) = g_signal^2;

% P=c2R(cs(1:n)) ;
P_state_estimate_covariance_matrix = lyap1(state_transition_matrix,Q_process_noise) ;

% initial uninformative guess
x_state_estimate = zeros(AR_model_order,1) ;
x_state_estimate_total_filtered1 = zeros(AR_model_order,input_signal_length) ;
x_state_estimate_total_filtered2 = zeros(AR_model_order,input_signal_length) ;
backward_state_transition_matrix = zeros(AR_model_order,AR_model_order) ;
backward_state_transition_matrix_tracked = zeros(input_signal_length,AR_model_order*AR_model_order) ;
y_signal_estimate = zeros(1,AR_model_order) ;



total_error=0 ;
%Forward run:
for i = 1:input_signal_length
    
    %get residual/prediction error and kalman gain:
    residual_error = input_signal(i) - x_state_estimate(AR_model_order);
    total_error = total_error + residual_error^2;
    kalman_gain = P_state_estimate_covariance_matrix*H_measurement_matrix' ...
        / (H_measurement_matrix*P_state_estimate_covariance_matrix*H_measurement_matrix'+R_noise_covariance);
    
    %use kalman gain to correct state estimate and state estimate covariance matrix:
    x_state_estimate = x_state_estimate + kalman_gain*residual_error;
    P_state_estimate_covariance_matrix = P_state_estimate_covariance_matrix ...
                                  - kalman_gain * H_measurement_matrix * P_state_estimate_covariance_matrix;
    
    %track filtered estimate:
    x_state_estimate_total_filtered1(:,i) = x_state_estimate; 
    
    %Forward state and covariance in time:
    x_state_estimate = state_transition_matrix * x_state_estimate;
    temp = P_state_estimate_covariance_matrix * state_transition_matrix';
    P_state_estimate_covariance_matrix = state_transition_matrix*temp + Q_process_noise;
    
    %keep track of state estimate for backward run:
    x_state_estimate_total_filtered2(:,i) = x_state_estimate;
    backward_state_transition_matrix = temp / P_state_estimate_covariance_matrix;
    backward_state_transition_matrix_tracked(i,:) = ...
        reshape(backward_state_transition_matrix,1,AR_model_order*AR_model_order);
end

%Backward run:
xl = x_state_estimate;
for i = input_signal_length:-1:1
    
    backward_state_transition_matrix = ...
        reshape(backward_state_transition_matrix_tracked(i,:),AR_model_order,AR_model_order);
    
    x_state_estimate = x_state_estimate_total_filtered1(:,i) ...
        + backward_state_transition_matrix * (xl-x_state_estimate_total_filtered2(:,i));
    
    xl = x_state_estimate;
    
    y_signal_estimate(i)=x_state_estimate(AR_model_order) ;
end



function [y_signal_estimate, total_forward_prediction_error] = ...
    get_clean_signal_kalman_filter_with_signal_and_noise_as_AR(z_input_signal,AR_parameters_signal,g_signal,...
                                auto_correlation_signal,AR_parameters_noise,g_noise,auto_correlation_noise)
% This function takes the corrupted signal z=signal+noise,
% AR parameters of signal (as,bs) and noise (an,bn)
% and signal resp. noise autocorrelations (cs,cn) and produces the smoothed
% estimate y of signal using bidirectional nonstationary Kalman filter
% te is the total forward prediction error
% it is optimized for AR models but not with respect to Matlab
% but rather with respect to the number of operations

%Get state space model from AR parameters, g, and auto-correlations:
[ A_state_transition_matrix, H_measurement_matrix, Q_process_noise_covariance_matrix, R_measurement_covariance, P_state_estimate_covariance_matrix] = ...
    AR_parameters_to_state_space_model(...
                                        AR_parameters_signal,...
                                        g_signal,...
                                        auto_correlation_signal,...
                                        AR_parameters_noise,...
                                        g_noise,...
                                        auto_correlation_noise) ;

%Get model lengths:
AR_model_order_signal = length(AR_parameters_signal) ;
AR_model_order_noise = length(AR_parameters_noise) ;
total_state_model_order = length(A_state_transition_matrix) ;
signal_length = length(z_input_signal) ;
 
%initial uninformative guess:
x_state_estimate = zeros(total_state_model_order,1) ;
x_state_estimate_total_filtered1 = zeros(total_state_model_order,signal_length) ;
x_state_estimate_total_filtered2 = zeros(total_state_model_order,signal_length) ;
backward_state_transition_matrix = zeros(total_state_model_order,total_state_model_order) ;
backward_state_transition_matrix_tracked = zeros(signal_length,total_state_model_order*total_state_model_order) ;
y_signal_estimate = zeros(1,total_state_model_order) ;


total_forward_prediction_error = 0;
%Forward run:
for i=1:signal_length
    
    %get prediction error:
    prediction_error = z_input_signal(i) - (x_state_estimate(AR_model_order_noise)+x_state_estimate(total_state_model_order));
    total_forward_prediction_error = total_forward_prediction_error + prediction_error^2 ;
    kalman_gain = P_state_estimate_covariance_matrix*H_measurement_matrix' ...
        / (H_measurement_matrix*P_state_estimate_covariance_matrix*H_measurement_matrix'+R_measurement_covariance) ;
    % L=(P(:,n)+P(:,nn)) / (P(n,n)+P(ns,ns)+2*P(ns,n)+r) ; % Kalman gain
    
    %get aposteriori state estimate:
    x_state_estimate = x_state_estimate + kalman_gain*prediction_error ;
    P_state_estimate_covariance_matrix = P_state_estimate_covariance_matrix - kalman_gain*H_measurement_matrix*P_state_estimate_covariance_matrix ;
        
    %Filtered estimate:
    x_state_estimate_total_filtered1(:,i) = x_state_estimate ;              
    
    %Forward state and covariance in time:
    x_state_estimate = A_state_transition_matrix*x_state_estimate ;
    temp = P_state_estimate_covariance_matrix * A_state_transition_matrix';
    P_state_estimate_covariance_matrix = A_state_transition_matrix*P_state_estimate_covariance_matrix*A_state_transition_matrix' ...
        + Q_process_noise_covariance_matrix ;
    
    %keep track of state estimate for backward run:
    x_state_estimate_total_filtered2(:,i) = x_state_estimate ;
    backward_state_transition_matrix = temp / P_state_estimate_covariance_matrix ;
    backward_state_transition_matrix_tracked(i,:) = reshape(backward_state_transition_matrix,1,total_state_model_order*total_state_model_order) ;
end

%Backward run smoothing:
x_last = x_state_estimate ;
for i=signal_length:-1:1,
  
    backward_state_transition_matrix = ...
        reshape(backward_state_transition_matrix_tracked(i,:),total_state_model_order,total_state_model_order);
    
    x_state_estimate = x_state_estimate_total_filtered1(:,i) ...
        + backward_state_transition_matrix*(x_last-x_state_estimate_total_filtered2(:,i));
    
    x_last = x_state_estimate;
    
    y_signal_estimate(i) = x_state_estimate(total_state_model_order);
end



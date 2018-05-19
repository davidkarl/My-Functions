function [y_signal_estimate,residual_error_over_time] = kalman_filter_classic_filtering(input_signal,...
    A_state_transition_matrix,H_measurement_matrix,Q_process_noise_covariance,R_measurement_noise_covariance)
% Performs Kalman filtering for input-less white-noise driven system
% with one output, system matrix A, output matrix C, process noise covariance
% matrix Q and measurement noise covariance r.
% z is the measured output and y is the filtered output
% Usage: [y e]=kalman(z,A,C,Q,r) 

state_vec_length = length(A_state_transition_matrix);
input_signal_length = length(input_signal);

% initial uninformative guess
P = dlyap1(A_state_transition_matrix,Q_process_noise_covariance); 
x_state_estimate = zeros(state_vec_length,1);

for i = 1:input_signal_length
    %Data step:
    residual_error_over_time(i) = input_signal(i) - H_measurement_matrix*x_state_estimate;          % prediction error
    kalman_gain = P*H_measurement_matrix' ...
        / (H_measurement_matrix*P*H_measurement_matrix'+R_measurement_noise_covariance); % Kalman gain
    x_state_estimate = x_state_estimate + kalman_gain*residual_error_over_time(i);             % correct the state estimate
    P = P - kalman_gain*H_measurement_matrix*P;
    
    y_signal_estimate(i) = H_measurement_matrix * x_state_estimate;
    %Time step:
    x_state_estimate = A_state_transition_matrix * x_state_estimate;
    P = A_state_transition_matrix*P*A_state_transition_matrix' + Q_process_noise_covariance;
end
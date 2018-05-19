function [y_signal_estimate,residual_error_over_time] = kalman_filter_filtering_bidirectional(...
           input_signal,A_state_trasition_matrix,H_measurement_matrix,...
           Q_process_noise_covariance,R_measurement_noise_covariance,P_state_estimate_covariance) 
% Performs bidirectional Kalman filtering for input-less white-noise driven 
% system with one output, system matrix A, output matrix C, process noise 
% covariance matrix Q, measurement noise covariance r and initial state
% estimate covariance P  
% z is the measured output and y is the estimated state (unlike kalmanr) 
% y(t,:) gives the state estimate at time t
% Usage: [y e]=kalmanrsi1(z,A,C,Q,r)

state_vector_length = length(A_state_trasition_matrix);
input_signal_length = length(input_signal);

% initial uninformative guess
x_state_estimate = zeros(state_vector_length,1);
x_state_estimate_filtered_over_time1 = zeros(state_vector_length,input_signal_length);
Pf = zeros(state_vector_length,state_vector_length);
x_state_estimate_filtered_over_time2 = zeros(state_vector_length,input_signal_length);
Pp = zeros(state_vector_length,state_vector_length);
Fp = zeros(input_signal_length,state_vector_length*state_vector_length);
y_signal_estimate = zeros(input_signal_length,state_vector_length);
residual_error_over_time = zeros(1,input_signal_length);

% Forward run
for i=1:input_signal_length
 %Data step
 residual_error_over_time(i) = input_signal(i) - H_measurement_matrix*x_state_estimate;          % prediction error
 kalman_gain = P_state_estimate_covariance*H_measurement_matrix' ...
     / (H_measurement_matrix*P_state_estimate_covariance*H_measurement_matrix' + R_measurement_noise_covariance) ; % Kalman gain
 x_state_estimate = x_state_estimate + kalman_gain*residual_error_over_time(i);             % correct the state estimate
 P_state_estimate_covariance = P_state_estimate_covariance - ...
                                    kalman_gain*H_measurement_matrix*P_state_estimate_covariance ;

 Pf = P_state_estimate_covariance;
 x_state_estimate_filtered_over_time1(:,i) = x_state_estimate;              % filtered estimate
 
 % Time step
 x_state_estimate = A_state_trasition_matrix*x_state_estimate ;
 P_state_estimate_covariance = A_state_trasition_matrix*P_state_estimate_covariance*A_state_trasition_matrix' ...
                                + Q_process_noise_covariance;

 x_state_estimate_filtered_over_time2(:,i) = x_state_estimate ;              % predicted estimate
 F = (Pf*A_state_trasition_matrix')/P_state_estimate_covariance ;
 Fp(i,:) = reshape(F,1,state_vector_length*state_vector_length) ;
end

%Backward run:
xl = x_state_estimate;
for i = input_signal_length:-1:1
  F = reshape(Fp(i,:),state_vector_length,state_vector_length);
  x_state_estimate = x_state_estimate_filtered_over_time1(:,i) + ...
                            F * (xl-x_state_estimate_filtered_over_time2(:,i));
  xl = x_state_estimate;
  y_signal_estimate(i,:) = x_state_estimate';
end



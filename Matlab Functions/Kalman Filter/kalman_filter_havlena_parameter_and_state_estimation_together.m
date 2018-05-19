function f_joint_estimate_over_time = kalman_filter_havlena_parameter_and_state_estimation_together(...
    input_signal,u,c)
% Implements Mr.Havlena's simultaneous parameter and state 
% Kalman filter estimator

%WHAT ARE u & c ?!?!?!?!!?1?

input_signal_length=length(input_signal) ;
n=length(c) ; % Assume the system to be of the order n


% Initial joint parameter and state vector fi=[b a x], b is n+1, 
% a is n and x is also n elements long
f_joint_estimate_current = zeros(1,3*n+1);
% We shall store it at every time point into the following array
f_joint_estimate_over_time = zeros(input_signal_length,3*n+1);

% Covariance matrix
P_state_estimate_covariance = 1e20*eye(3*n+1) ;

% Here the estimation begins
for i=1:input_signal_length,
 %for each sample:
 h_measurement_matrix = [u(i) , zeros(1,2*n) , 1 , zeros(1,n-1)] ;
 %calculate Kalman gain:
 kalman_gain = h_measurement_matrix*P_state_estimate_covariance / ...
                (1+h_measurement_matrix*P_state_estimate_covariance*h_measurement_matrix');
 %calculate the output error:
 residual_error = input_signal(i) - h_measurement_matrix*f_joint_estimate_current';
 % perform the data step - i.e. improve the estimate
 f_joint_estimate_current = f_joint_estimate_current + kalman_gain*residual_error;
 P_state_estimate_covariance = P_state_estimate_covariance - ...
     (P_state_estimate_covariance*h_measurement_matrix'*h_measurement_matrix*P_state_estimate_covariance) ...
     / (1 + h_measurement_matrix*P_state_estimate_covariance*h_measurement_matrix') ;
 % save the estimate
 f_joint_estimate_over_time(i,:) = f_joint_estimate_current;
 % perform the time step
 f_joint_estimate_current(2*n+2:3*n) = f_joint_estimate_current(2*n+3:3*n+1); 
 f_joint_estimate_current(3*n+1) = 0;
 f_joint_estimate_current(2*n+2:3*n+1) = f_joint_estimate_current(2*n+2:3*n+1) - ...
     input_signal(i)*f_joint_estimate_current(n+2:2*n+1) + u(i)*f_joint_estimate_current(2:n+1) ...
     - residual_error*c ;
end ;
 
 
 
function [output_signal,residual_error_vec,total_error] = ...
    ARX_model_kalman_filter_output_error(input_signal,AR_parameters,q_process_noise,r_measurement_noise)
% Perform Kalman filter for ARX model and known parameters a
% q is the system noise covariance and r the output noise covariance
% k is the steady-state Kalman gain row vector k1,...,kn 
% Usage: [ys k]=arxoekal(y,a,q,r)

%Initialize variables:
AR_model_order = length(AR_parameters); 
input_signal_length = length(input_signal);
output_signal = zeros(1,input_signal_length);
output_signal(1:AR_model_order) = input_signal(1:AR_model_order);
residual_error_vec = zeros(1,input_signal_length);

%Initialize state transition matrix and covariance matrix:
A_state_transition_matrix = zeros(AR_model_order);
A_state_transition_matrix(1,1:AR_model_order) = AR_parameters;
for i = 2:AR_model_order 
    A_state_transition_matrix(i,i-1) = 1; 
end
P_covariance = 5*eye(AR_model_order)*r_measurement_noise;

%Initialize state estimate vec:
state_estimate_vec = input_signal(1)*ones(AR_model_order,1);
for i=1:AR_model_order 
    state_estimate_vec(AR_model_order+1-i) = input_signal(i); 
end

%the first time step:
state_estimate_vec = A_state_transition_matrix*state_estimate_vec;

total_error = 0;
for i=AR_model_order+1:input_signal_length
   %store the predicted estimate:
   output_signal(i) = state_estimate_vec(1);
   
   %make the data step:
   residual_error_vec(i) = input_signal(i)-state_estimate_vec(1) ;
   total_error = total_error + residual_error_vec(i)^2 ;
   state_estimate_vec = state_estimate_vec + P_covariance(:,1)/(P_covariance(1,1)+r_measurement_noise)*residual_error_vec(i);

   %perform the time step:
   state_estimate_vec = A_state_transition_matrix*state_estimate_vec;
   P_covariance = A_state_transition_matrix...
       *( P_covariance - P_covariance(:,1)*P_covariance(1,:)/(P_covariance(1,1)+r_measurement_noise) )...
       *A_state_transition_matrix';
   P_covariance(1,1) = P_covariance(1,1) + q_process_noise;
end

Kalman_gain_final = (P_covariance(:,1)/(P_covariance(1,1)+r))';



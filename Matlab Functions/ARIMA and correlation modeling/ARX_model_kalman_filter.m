function [output_signal] = ARX_model_kalman_filter(input_signal,AR_parameters,q_noise_covariance)
% Perform Kalman filter for ARX model and known parameters a
% q is the noise covariance and P the initial covariance matrix
% estimate

AR_model_order = size(AR_parameters,2); 
input_signal_length = size(input_signal,2);
output_signal = zeros(1,input_signal_length);
output_signal(1:AR_model_order+1) = input_signal(1:AR_model_order+1);

%Get state transition matrix and state covariance matrix:
A_state_transition_matrix = zeros(AR_model_order+1);
A_state_transition_matrix(1,1:AR_model_order) = AR_parameters;
for i=2:AR_model_order+1 
    A_state_transition_matrix(i,i-1)=1; 
end
P_covariance = eye(AR_model_order+1) * q_noise_covariance ;

%Initialize state vector:
x = zeros(AR_model_order+1,1);
for i=1:AR_model_order 
    x(AR_model_order+2-i) = input_signal(i); 
end

%the first time step:
x = A_state_transition_matrix*x;

for i=AR_model_order+1:input_signal_length
   %store the estimate:
   output_signal(i) = x(1);
   
   %make the data step:
   x = x + P_covariance(:,1)/P_covariance(1,1)*(input_signal(i)-x(1));
   
   %perform the time step:
   x = A_state_transition_matrix*x ;
   P_covariance = A_state_transition_matrix...
                    *( P_covariance - P_covariance(:,1)*P_covariance(1,:)/P_covariance(1,1) )... 
                    * A_state_transition_matrix';
   P_covariance(1,1) = P_covariance(1,1) + q_noise_covariance;
end  
 



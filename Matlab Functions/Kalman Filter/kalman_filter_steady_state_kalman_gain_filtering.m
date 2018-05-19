function [y_signal_estimate,residual_error_over_time] = kalman_filter_steady_state_kalman_gain_filtering(...
                input_signal, A_steady_state_state_transition_matrix, kalman_gain_stead_state)
% Performs steady-state Kalman filtering
% assumes z=ys+e, where e is the innovation and ys is generated
% by the x(t+1)=Ax(t), ys=[0..0 1] x discrete system. 
% k is the Kalman gain
% Usage: [ys,e]=kalmk(z,A,k)
% ys,e,z,k are row vectors

input_signal_length = length(input_signal);
state_vector_length = length(A_steady_state_state_transition_matrix);
kalman_gain_stead_state = kalman_gain_stead_state';

x_estimate = input_signal(1)*zeros(state_vector_length,1) ; % initial state estimate mean

for i=1:input_signal_length
    y_signal_estimate(i) = x_estimate(state_vector_length) ;         % prediction of y
    residual_error_over_time(i) = input_signal(i)-y_signal_estimate(i)   ;  % the inovation
    x_estimate = x_estimate + kalman_gain_stead_state*residual_error_over_time(i)        ;  % make the data step
    y_signal_estimate(i) = x_estimate(state_vector_length) ;         % estimation of y
    x_estimate = A_steady_state_state_transition_matrix*x_estimate ;              % make the time step
end

 
 






function [y_signal_estimate,residual_error_over_time,g_error_gradient_over_time,kalman_gain_final] = ...
                kalman_filter_steady_state_filtering_AR_parameters(input_signal,AR_parameters_signal,kalman_gain)
% Performs steady-state Kalman filtering
% assumes z=ys+e, where e is the innovation and ys is generated
% by the x(t+1)=Ax(t), ys=[0..0 1] x discrete system. 
% k is the Kalman gain
% Usage: [ys,e]=kalmkw(z,A,k)
% ys,e,z,k are row vectors
%
% Kalman gain improvements by gradient algorithm
gamma=1e-9 ;

input_signal_length = length(input_signal);
AR_model_order = length(AR_parameters_signal);
A_state_transition_matrix = AR_parameters_to_state_transfer_matrix(AR_parameters_signal);
kalman_gain = kalman_gain';

%EXPAND THIS FUNCTION TO GENERAL AR MODEL!!!!!!!!
if AR_model_order ~= 4 
    error('So far, kalmkw works only for n=4');
end

x = input_signal(1)*ones(AR_model_order,1) ; % initial state estimate mean
x = zeros(AR_model_order,1) ;
g_error_gradient_over_time = zeros(input_signal_length,AR_model_order) ;
total_error = 0;

for i=1:input_signal_length
    y_signal_estimate(i) = x(AR_model_order);         % prediction of y
    residual_error_over_time(i) = input_signal(i) - y_signal_estimate(i);  % the inovation
    x = x + kalman_gain*residual_error_over_time(i);  % make the data step
    total_error = total_error + residual_error_over_time(i)^2;
    
    % the gradients
    if i>AR_model_order
        g_error_gradient_over_time(i,4) = -2*residual_error_over_time(i)*...
            residual_error_over_time(i-1) ;
        
        g_error_gradient_over_time(i,3) = -2*residual_error_over_time(i)*...
            (AR_parameters_signal(2)*residual_error_over_time(i-2)+AR_parameters_signal(3)*residual_error_over_time(i-3)+AR_parameters_signal(4)*residual_error_over_time(i-4)) ;
        
        g_error_gradient_over_time(i,2) = -2*residual_error_over_time(i)*...
            (AR_parameters_signal(3)*residual_error_over_time(i-2)+AR_parameters_signal(4)*residual_error_over_time(i-3)) ;
        
        g_error_gradient_over_time(i,1) = -2*residual_error_over_time(i)*...
            AR_parameters_signal(4)*residual_error_over_time(i-2) ;
        
        % change the Kalman gain
        kalman_gain = kalman_gain + gamma*g_error_gradient_over_time(i,:)' ;
    end
    
    y_signal_estimate(i) = x(AR_model_order) ;         % estimation of y
    x = A_state_transition_matrix*x ;              % make the time step
end
kalman_gain_final = kalman_gain' ;
 




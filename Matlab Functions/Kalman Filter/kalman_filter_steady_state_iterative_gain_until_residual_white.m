function [y_signal_estimate,residual_error_over_time] = ...
                kalman_filter_steady_state_iterative_gain_until_residual_white(...
                                    input_signal,A_state_transition_matrix,kalman_gain)
% Performs steady-state Kalman filtering
% assumes z=ys+e, where e is the innovation and ys is generated
% by the x(t+1)=Ax(t), ys=[0..0 1] x discrete system. 
% Iteratively estimates the Kalman gain k and calls kalmk
% k is the first estimate of the gain
% Usage: [ys,e]=kalmke(z,A,k)
% ys,e,z,k are row vectors

input_signal_length = length(input_signal) ;
state_vector_length = length(A_state_transition_matrix) ;
kalman_gain = kalman_gain';

h_measurement_matrix = [zeros(1,state_vector_length-1) , 1]; % output vector h 

[y_signal_estimate,residual_error_over_time] = ...
    kalman_filter_steady_state_kalman_gain_filtering(input_signal,A_state_transition_matrix,kalman_gain');  % estimate the output
desgree_of_whitness = get_degree_of_signal_whitness(residual_error_over_time); 

iterations_counter = 0;
while desgree_of_whitness<1  % as long as the innovation is not white
    iterations_counter = iterations_counter+1;
    if iterations_counter>10
        disp('kalmke: maximum (10) number of iteration reached.') ;
        break;
    end
    
    residual_error_auto_correlation = ...
        get_auto_correlation_using_direct_calculation( residual_error_over_time,state_vector_length+1 ); %autocorrelation of the inovation sequence
    
    residual_error_auto_correlation_partial_normalized = ...
        residual_error_auto_correlation(2:state_vector_length+1)/residual_error_auto_correlation(1);
    
    %construct the matrix q:
    aikh = A_state_transition_matrix * (eye(state_vector_length) - kalman_gain*h_measurement_matrix);
    it = eye(state_vector_length);
    q = zeros(state_vector_length);
    for j=1:state_vector_length
        q(j,:) = h_measurement_matrix*it*A_state_transition_matrix ;
        it = it*aikh;
    end
    % Compute new gain
    kalman_gain = kalman_gain + q \ residual_error_auto_correlation_partial_normalized ;
    [y_signal_estimate , residual_error_over_time] = ...
       kalman_filter_steady_state_kalman_gain_filtering(input_signal,A_state_transition_matrix,kalman_gain');
   
   %Check whitness:
   desgree_of_whitness = get_degree_of_signal_whitness(residual_error_over_time); 
end






function [y_signal_estimate, total_error] = get_clean_signal_kalman_filter_with_signal_and_noise_as_AR2(input_signal,...
                            AR_parameters_signal,g_signal,auto_correlation_signal,...
                            AR_parameters_noise,g_noise,auto_correlation_noise)
% This function takes the corrupted signal z=signal+noise,
% AR parameters of signal (as,bs) and noise (an,bn)
% and signal resp. noise autocorrelations (cs,cn) and produces the smoothed
% estimate y of signal using bidirectional nonstationary Kalman filter
% te is the total forward prediction error
% it is optimized for AR models but not with respect to Matlab
% but rather with respect to the number of operations

[A_state_transition_matrix,H_measurement_matrix,Q_process_noise_covariance,R_measurement_covariance,P_state_estimate_covariance] = ...
    get_state_space_model_using_signal_and_noise_AR_parameters(...
                                        AR_parameters_signal,...
                                        g_signal,...
                                        auto_correlation_signal,...
                                        AR_parameters_noise,...
                                        g_noise,...
                                        auto_correlation_noise); 
                                    

AR_model_order_signal = length(AR_parameters_signal);
AR_model_order_noise = length(AR_parameters_noise);
state_vector_length = length(A_state_transition_matrix);
input_signal_length = length(input_signal);

%initial uninformative guess:
x_state_estimate = zeros(state_vector_length,1);
x_state_estimate_total_filtered1 = zeros(state_vector_length,input_signal_length);
x_state_estimate_total_filtered2 = zeros(state_vector_length,input_signal_length);
backward_state_transition_matrix = zeros(state_vector_length,state_vector_length);
backward_state_transition_matrix_tracked = zeros(input_signal_length,state_vector_length*state_vector_length);
y_signal_estimate = zeros(1,state_vector_length);



total_error = 0;

%Forward run:
for i = 1:input_signal_length
    %Data step:
    residual_error = input_signal(i)-(x_state_estimate(AR_model_order_noise)+x_state_estimate(state_vector_length));          % prediction error
    total_error = total_error + residual_error^2 ;
    kalman_gain = P_state_estimate_covariance*H_measurement_matrix' / ...
        (H_measurement_matrix*P_state_estimate_covariance*H_measurement_matrix'+R_measurement_covariance);
    
    x_state_estimate = x_state_estimate + kalman_gain*residual_error;
    
    P_state_estimate_covariance = P_state_estimate_covariance - ...
        kalman_gain*H_measurement_matrix*P_state_estimate_covariance;
    
    x_state_estimate_total_filtered1(:,i) = x_state_estimate;
    
    x_state_estimate = A_state_transition_matrix*x_state_estimate;
    
    PfA=P_state_estimate_covariance*A_state_transition_matrix' ;
    P_state_estimate_covariance=A_state_transition_matrix*PfA+Q_process_noise_covariance ;
    
    x_state_estimate_total_filtered2(:,i)=x_state_estimate ;              % predicted estimate
    backward_state_transition_matrix=PfA/P_state_estimate_covariance ;
    backward_state_transition_matrix_tracked(i,:)=reshape(backward_state_transition_matrix,1,state_vector_length*state_vector_length) ;
end

% Backward run
xl=x_state_estimate ;
for i=input_signal_length:-1:1,
  backward_state_transition_matrix=reshape(backward_state_transition_matrix_tracked(i,:),state_vector_length,state_vector_length) ;
  x_state_estimate=x_state_estimate_total_filtered1(:,i)+backward_state_transition_matrix*(xl-x_state_estimate_total_filtered2(:,i)) ;
  xl=x_state_estimate ;
  y_signal_estimate(i)=x_state_estimate(state_vector_length) ;
end ;



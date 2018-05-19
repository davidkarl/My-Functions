function [y_signal_estimate,residual_error_over_time,kalman_gain_final] = ...
                kalman_filter_steady_state_iterative_until_steady_gain(...
                                        input_signal,A_state_transition_matrix,kalman_gain)
% Performs steady-state Kalman filtering
% assumes z=ys+e, where e is the innovation and ys is generated
% by the x(t+1)=Ax(t), ys=[0..0 1]*x discrete system. 
% Iteratively estimates the Kalman gain k and calls kalmk
% k is the first estimate of the gain. kout is the resulting gain
% Tries to use more autocorrelation coefficients than n, currently 20.
% Usage: [ys,e,kout]=kalmkee(z,A,k)
% ys,e,z,k,kout are row vectors
% 
% This function performs generally better than kalmke. It does not use the
% autocorrelation function to determine the whiteness of the inovation
% sequence, instead it filters until convergence (of k) is attained.
% Usually, this is faster.

input_signal_length = length(input_signal);
state_vector_length = length(A_state_transition_matrix);
kalman_gain = kalman_gain';
number_of_auto_correlation_coefficients = max(200,state_vector_length); % number of autocorr coefs used
precision_tolerance = 1e-2;   % the desired precision

%Fix for unstable matrixes:
while (max(abs(roots(poly(A_state_transition_matrix))))>0.99) 
    A_state_transition_matrix = 0.99*A_state_transition_matrix; 
end


H_measurement_matrix = [zeros(1,state_vector_length-1) , 1]; % output vector h 

[y_signal_estimate , residual_error_over_time] = kalman_filter_steady_state_kalman_gain_filtering(...
                                                        input_signal,A_state_transition_matrix,kalman_gain');  % estimate the output

iterations_counter = 0;
kalman_gain_previous = 10*kalman_gain + 1e6 * ones(state_vector_length,1);
relative_change_in_kalman_gain = 1; %something larger the precision_tolerances

while relative_change_in_kalman_gain > precision_tolerance
  
  iterations_counter = iterations_counter+1;
  kalman_gain_previous = kalman_gain;
  
  if iterations_counter > 100 
    disp('kalmkee: maximum (100) number of iteration reached.') ;
    break;
  end
  
  auto_correlation_of_residual_error = get_auto_correlation_using_power_spectrum_and_ifft(...
                        residual_error_over_time , number_of_auto_correlation_coefficients+1) ; % autocorrelation of the inovation sequence

  auto_correlation_of_residual_error_normalized = ...
      auto_correlation_of_residual_error(2:number_of_auto_correlation_coefficients+1)/auto_correlation_of_residual_error(1) ; 
  
  %construct the matrix q
  aikh = A_state_transition_matrix * (eye(state_vector_length) - kalman_gain*H_measurement_matrix); 
  it = eye(state_vector_length);
  q = zeros(state_vector_length); 
  for j=1:number_of_auto_correlation_coefficients
     q(j,:) = H_measurement_matrix * it * A_state_transition_matrix;
     it = it*aikh;
  end
  % Compute new gain
  kalman_gain = kalman_gain+q \ auto_correlation_of_residual_error_normalized ;   % \ represents LSM solution


  max_root = max(abs(roots(poly(A_state_transition_matrix*(eye(n)-kalman_gain*H_measurement_matrix)))));
  while max_root >= 1
      k = 0.1*abs(k);
      max_root = max(abs(roots(poly(A*(eye(n)-k*h)))));
  end	
   
   for ki = 1:state_vector_length
       if isnan(kalman_gain(ki))
           disp('Divergence in k detected, aborting') ;
           kalman_gain = zeros(state_vector_length,1) ;
           break;
       end
   end


  [y_signal_estimate , residual_error_over_time] = kalman_filter_steady_state_kalman_gain_filtering(...
                                                      input_signal,A_state_transition_matrix,kalman_gain');
 
   relative_change_in_kalman_gain = sum(abs(kalman_gain_previous-kalman_gain))/sum(abs(kalman_gain)+1e-6);
end

kalman_gain_final = kalman_gain' ;     

















function [y_signal_estimate] = kalman_filter_sound_filtering_example1(input_signal)
% Filters a sound signal 
% based on Kalman filtration
% Usage: y=fltsnd(x), where x is the signal to be filtered (row vector)
% Jan Kybic, 14.3.1997
% changed 22.4.1997
% changed 1.6.1997

samples_per_frame = 256; 	    % window length
AR_model_order = 4; 		    % order of the system	
kalman_gain_initial = 0.1*ones(1,AR_model_order);   % initial Kalman gain
kalman_gain_final = kalman_gain_initial;
input_signal_length = length(input_signal);
hanning_window = hanning(samples_per_frame)';
y_signal_estimate = zeros(1,input_signal_length);      % output vector

input_signal = input_signal - mean(input_signal); %get rid of DC
dpx = sum(input_signal.*input_signal);

% We use w/2 overlap:
number_of_frames = floor(input_signal_length/(samples_per_frame/2))-2 ;

for frame_counter = 0:number_of_frames % l is the window index
   
  start_index = frame_counter*(samples_per_frame/2) + 1; 
  stop_index = frame_counter*(samples_per_frame/2) + samples_per_frame;
  
  current_frame = input_signal(start_index:stop_index); % get one window
  
  AR_parameters = get_AR_parameters_repetitive_with_weights(current_frame,AR_model_order); % find the AR model parameters
  A_state_transition_matrix = AR_parameters_to_state_transfer_matrix(AR_parameters) ;
  
  %Fix for unstable matrixes
  while (max(abs(roots(poly(A_state_transition_matrix))))>0.999) 
      A_state_transition_matrix = 0.99*A_state_transition_matrix; 
  end

  % Filter, adaptively find the correct Kalman gain
  % Unfortunately, we may not use the old values
  [current_frame ,residual_error_over_time ,kalman_gain_final] = ...
                kalman_filter_steady_state_iterative_until_steady_gain(...
                            current_frame,A_state_transition_matrix,kalman_gain_initial) ;
  
  current_frame_windowed = current_frame .* hanning_window ;
  
  if frame_counter==1
      current_frame_windowed = [current_frame(1:samples_per_frame/2) , ...
                                    current_frame_windowed(samples_per_frame/2+1:samples_per_frame)];
  elseif frame_counter == number_of_frames
      current_frame_windowed = [current_frame_windowed(1:samples_per_frame/2) , ...
                                        current_frame(samples_per_frame/2+1:samples_per_frame)];
  end
  
  y_signal_estimate(start_inde:stop_index) = y_signal_estimate(start_inde:stop_index) + ...
                                                                            current_frame_windowed;
end

% Adjust the mean to zero 
y_signal_estimate = y_signal_estimate - mean(y_signal_estimate) ;
     	






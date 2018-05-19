function [y_signal_estimate] = get_clean_signal_kalman_filter_with_signal_and_noise_as_AR4(...
    input_signal,AR_parameters_signal,g_signal,auto_correlation_signal,...
    AR_parameters_noise,g_noise,auto_correlation_noise)
% This function takes the corrupted signal z=signal+noise,
% AR parameters of signal (as,bs) and noise (an,bn)
% and signal resp. noise autocorrelations (cs,cn) and produces the smoothed
% estimate y of signal using bidirectional nonstationary Kalman filter
% Usage: y=darkalm3(z,as,bs,cs,an,bn,cn) ;

% it is optimized for AR models but not with respect to Matlab
% but rather with respect to the number of operations

AR_model_order_signal = length(AR_parameters_signal);
AR_model_order_noise = length(AR_parameters_noise);
AR_model_order_signal_plus_noise = AR_model_order_signal + AR_model_order_noise;

A_state_transition_matrix = zeros(AR_model_order_signal_plus_noise); % system transfer matrix
P_state_estimate_covariance = zeros(AR_model_order_signal_plus_noise); % initial state estimate covariance matrix
H_measurement_matrix = [zeros(1,AR_model_order_noise-1) , 1 , zeros(1,AR_model_order_signal-1) ,1]; % output matrix (vector, actually)

A_state_transition_matrix(1:AR_model_order_noise,1:AR_model_order_noise) = ...
                                    AR_parameters_to_state_transfer_matrix(AR_parameters_noise);
A_state_transition_matrix(AR_model_order_noise+1:AR_model_order_signal_plus_noise,AR_model_order_noise+1:AR_model_order_signal_plus_noise) = ...
                                    AR_parameters_to_state_transfer_matrix(AR_parameters_signal);
                                
Q_process_noise_covariance_noise = g_noise^2;
Q_process_noise_covariance_signal = g_signal^2;

R_measurement_noise_covariance = (Q_process_noise_covariance_noise + Q_process_noise_covariance_signal)*1e-6;

P_state_estimate_covariance(1:AR_model_order_noise,1:AR_model_order_noise) = ...
                    auto_correlation_to_toeplitz_matrix(auto_correlation_noise(1:AR_model_order_noise));
P_state_estimate_covariance(AR_model_order_noise+1:AR_model_order_signal_plus_noise,AR_model_order_noise+1:AR_model_order_signal_plus_noise) = ...
                    auto_correlation_to_toeplitz_matrix(auto_correlation_signal(1:AR_model_order_signal));

AR_model_order_signal_plus_noise = length(A_state_transition_matrix) ;
input_signal_length = length(input_signal);

%initial uninformative guess
x_state_estimate = zeros(AR_model_order_signal_plus_noise,1);
x_state_estimate_total_filtered1 = zeros(AR_model_order_signal_plus_noise,input_signal_length);
Pp = zeros(input_signal_length,AR_model_order_signal_plus_noise*AR_model_order_signal_plus_noise);
kalman_gain_over_time = zeros(input_signal_length,AR_model_order_signal_plus_noise);
pomp = zeros(input_signal_length);
y_signal_estimate = zeros(1,AR_model_order_signal_plus_noise);
residual_error_over_time = zeros(input_signal_length);
Almb = zeros(AR_model_order_signal_plus_noise,1);



%Forward run:
for i=1:input_signal_length,
    %Data step:
    % e=z(i)-C*x ;
    residual_error = input_signal(i)-(x_state_estimate(AR_model_order_noise)+x_state_estimate(AR_model_order_signal_plus_noise)) ;          % prediction error
    residual_error_over_time(i) = residual_error;
    
    % K=C' / (C*P*C'+r) ;
    % L=P*K ;
    pom = 1 / ( P_state_estimate_covariance(AR_model_order_signal_plus_noise,AR_model_order_signal_plus_noise) ...
              + P_state_estimate_covariance(AR_model_order_signal,AR_model_order_signal) ...
              + 2*P_state_estimate_covariance(AR_model_order_signal,AR_model_order_signal_plus_noise ) ... 
              + R_measurement_noise_covariance);
          
    pomp(i) = pom;
    K = H_measurement_matrix'*pom ;
    kalman_gain = (P_state_estimate_covariance(:,AR_model_order_signal_plus_noise)+P_state_estimate_covariance(:,AR_model_order_noise)) ...
            * pom; % Kalman gain
    kalman_gain_over_time(i,:) = kalman_gain';
    x_state_estimate = x_state_estimate + kalman_gain*residual_error ;             % correct the state estimate
    
    % P=P-L*C*P ;
    P_state_estimate_covariance = P_state_estimate_covariance -kalman_gain * ( ...
        P_state_estimate_covariance(AR_model_order_signal_plus_noise,:) + P_state_estimate_covariance(AR_model_order_noise,:));
    
    x_state_estimate_total_filtered1(:,i) = x_state_estimate;              % filtered estimate
    Pp(i,:) = reshape(P_state_estimate_covariance,1,AR_model_order_signal_plus_noise*AR_model_order_signal_plus_noise) ;
    
    % Time step
    % x=A*x ;
    xnew = AR_parameters_signal(AR_model_order_signal:-1:1) * x_state_estimate(AR_model_order_noise+1:AR_model_order_signal_plus_noise);
    x_state_estimate(AR_model_order_noise+1:AR_model_order_signal_plus_noise-1) = x_state_estimate(AR_model_order_noise+2:AR_model_order_signal_plus_noise);
    x_state_estimate(AR_model_order_signal_plus_noise) = xnew;
    xnew = AR_parameters_noise(AR_model_order_noise:-1:1)*x_state_estimate(1:AR_model_order_noise);
    x_state_estimate(1:AR_model_order_noise-1) = x_state_estimate(2:AR_model_order_noise);
    x_state_estimate(AR_model_order_noise) = xnew;
    
    % P=A*P*A'+Q ;
    % P=A*P*A' ;
    for ii = 1:AR_model_order_signal_plus_noise
        xnew = AR_parameters_signal(AR_model_order_signal:-1:1) * ...
                    P_state_estimate_covariance(AR_model_order_noise+1:AR_model_order_signal_plus_noise,ii) ;
        
        P_state_estimate_covariance(AR_model_order_noise+1:AR_model_order_signal_plus_noise-1,ii) = ...
                    P_state_estimate_covariance(AR_model_order_noise+2:AR_model_order_signal_plus_noise,ii) ;
        
        P_state_estimate_covariance(AR_model_order_signal_plus_noise,ii) = xnew ;
        
        xnew = AR_parameters_noise(AR_model_order_noise:-1:1) * ...
                    P_state_estimate_covariance(1:AR_model_order_noise,ii) ;
        
        P_state_estimate_covariance(1:AR_model_order_noise-1,ii) = ...
                    P_state_estimate_covariance(2:AR_model_order_noise,ii) ;
        
        P_state_estimate_covariance(AR_model_order_noise,ii) = xnew;
    end
    
    for ii = 1:AR_model_order_signal_plus_noise
        xnew = AR_parameters_signal(AR_model_order_signal:-1:1) * ...
                    P_state_estimate_covariance(ii,AR_model_order_noise+1:AR_model_order_signal_plus_noise)';
        
        P_state_estimate_covariance(ii,AR_model_order_noise+1:AR_model_order_signal_plus_noise-1) = ...
                    P_state_estimate_covariance(ii,AR_model_order_noise+2:AR_model_order_signal_plus_noise);
        
        P_state_estimate_covariance(ii,AR_model_order_signal_plus_noise) = xnew;
        
        xnew = AR_parameters_noise(AR_model_order_noise:-1:1) * ...
                    P_state_estimate_covariance(ii,1:AR_model_order_noise)';
       
        P_state_estimate_covariance(ii,1:AR_model_order_noise-1) = ...
                    P_state_estimate_covariance(ii,2:AR_model_order_noise);
        
        P_state_estimate_covariance(ii,AR_model_order_noise) = xnew;
    end 
    
    P_state_estimate_covariance(AR_model_order_signal_plus_noise,AR_model_order_signal_plus_noise) = ...
          P_state_estimate_covariance(AR_model_order_signal_plus_noise,AR_model_order_signal_plus_noise) + ...
          Q_process_noise_covariance_signal;
    
      P_state_estimate_covariance(AR_model_order_noise,AR_model_order_noise) = ...
          P_state_estimate_covariance(AR_model_order_noise,AR_model_order_noise) + ...
          Q_process_noise_covariance_noise;
end 

% Backward run
lmb = zeros(AR_model_order_signal_plus_noise,1); 
y_signal_estimate(input_signal_length) = ...
    x_state_estimate_total_filtered1(AR_model_order_signal_plus_noise,input_signal_length);

for i = input_signal_length:-1:1,
    kalman_gain = kalman_gain_over_time(i,:)';
    P_state_estimate_covariance = reshape(Pp(i,:),AR_model_order_signal_plus_noise,AR_model_order_signal_plus_noise) ;
    
    % Almb=A'*lmb ;
    Almb(1) = lmb(AR_model_order_noise)*AR_parameters_noise(AR_model_order_noise);
    Almb(2:AR_model_order_noise) = lmb(1:AR_model_order_noise-1) + ...
        lmb(AR_model_order_noise).*AR_parameters_noise(AR_model_order_noise-1:-1:1)';
    Almb(AR_model_order_noise+1) = lmb(AR_model_order_signal_plus_noise) * ...
                                        AR_parameters_signal(AR_model_order_signal);
    Almb(AR_model_order_noise+2 : AR_model_order_signal_plus_noise) = ...
        lmb(AR_model_order_noise+1:AR_model_order_signal_plus_noise-1) + ...
        lmb(AR_model_order_signal_plus_noise).*AR_parameters_signal(AR_model_order_signal-1:-1:1)';
    
    PAlmb = P_state_estimate_covariance * Almb;
    
    x_state_estimate=x_state_estimate_total_filtered1(:,i)+PAlmb ;
    
    y_signal_estimate(i)=x_state_estimate(AR_model_order_signal_plus_noise) ;
    
    Ls = kalman_gain(AR_model_order_signal_plus_noise) + kalman_gain(AR_model_order_noise); 
    Lr = Ls/R_measurement_noise_covariance;
    
    lmb = Almb; 
    g = (pomp(i)+Lr) * ((1-Ls)*residual_error_over_time(i) - PAlmb(AR_model_order_signal_plus_noise) - ...
                                                                PAlmb(AR_model_order_noise));
    lmb(AR_model_order_signal_plus_noise) = lmb(AR_model_order_signal_plus_noise) + g; 
    lmb(AR_model_order_noise) = lmb(AR_model_order_noise) + g ;
end ;






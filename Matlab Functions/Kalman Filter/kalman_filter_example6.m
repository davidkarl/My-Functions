% Demonstrates relative performance of Kalman filter 
% and Rauch-Tung-Striebel smoother on random walk estimation 

clear all; 
close all; 
N = 100;  % Number of samples of process used in simulations 
process_noise_variance = 0.01;  % Variance of random walk increments 
measurement_noise_variance = 1;    % Variance of sampling noise 
sigma_proces_noise  = sqrt(process_noise_variance); % Standard deviations 
sigma_measurement_noise  = sqrt(measurement_noise_variance); 

%
P_estimation_uncertainty_covariance_minus      = 100;           % Covariance of initial uncertainty 
x_true_state(1)    = sqrt(P_estimation_uncertainty_covariance_minus)*randn; % Initial value of true process 
x_estimated_state_minus(1)   = 0;             % Initial (predicted) value of true process 
sawtooth   = sqrt(P_estimation_uncertainty_covariance_minus); 
ts         = 0; 
% 

% Forward pass: filter 
for k=1:N; 
   t_vec(k) = k-1; 
   if k~=1 
      x_true_state(k)  = x_true_state(k-1) + sigma_proces_noise*randn;       % Random walk 
      P_estimation_uncertainty_covariance_minus(k) = P_estimation_uncertainty_covariance_plus(k-1) + process_noise_variance; 
      sawtooth = [sawtooth,sqrt(P_estimation_uncertainty_covariance_minus(k))]; 
      ts       = [ts,t_vec(k)]; 
      x_estimated_state_minus(k) = x_estimated_state_plus(k-1); 
   end; 
   z(k)     = x_true_state(k) + sigma_measurement_noise*randn;            % Noisy sample 
   K        = P_estimation_uncertainty_covariance_minus(k)/(P_estimation_uncertainty_covariance_minus(k)+measurement_noise_variance); 
   x_estimated_state_plus(k) = x_estimated_state_minus(k) + K*(z(k) - x_estimated_state_minus(k));  % Kalman filter estimate 
   P_estimation_uncertainty_covariance_plus(k) = P_estimation_uncertainty_covariance_minus(k) - K*P_estimation_uncertainty_covariance_minus(k); 
   sawtooth = [sawtooth,sqrt(P_estimation_uncertainty_covariance_plus(k))]; 
   ts       = [ts,t_vec(k)]; 
end; 

% Backward pass: smooth 
xsmooth = x_estimated_state_plus; 
for k=N-1:-1:1, 
   A          = P_estimation_uncertainty_covariance_plus(k)/P_estimation_uncertainty_covariance_minus(k+1); 
   xsmooth(k) = xsmooth(k) + A*(xsmooth(k+1) - x_estimated_state_minus(k+1)); 
end; 
plot(t_vec,x_true_state,'b-',t_vec,x_estimated_state_minus,'g:',t_vec,x_estimated_state_plus,'k-.',t_vec,xsmooth,'r--'); 
legend('True','Predicted','Corrected','Smoothed'); 
title('DEMO #7: Kalman Filter versus Rauch-Tung-Striebel Smoother'); 
xlabel('Discrete Time'); 
ylabel('Random Walk'); 
figure; 
semilogy(ts,sawtooth); 
xlabel('Discrete Time'); 
ylabel('RMS Estimation Uncertainty'); 
title('''Sawtooth Curve'''); 
 
function [AR_parameters,Q_process_noise_std,R_measurement_noise_std] = ...
    get_ARX_parameters_including_process_and_measurement_noise(input_signal,AR_model_order)
% Identifies ARX model of the n-th order using data y
% uses LSM - one time identification
% returns vector of n a coefficients
% q is the process noise dispersion estimate
% r is the measurement noise dispersion estimate

input_signal_length = length(input_signal);
z = zeros(input_signal_length-AR_model_order,AR_model_order);
b = input_signal(AR_model_order+1:input_signal_length) ;

for i=1:input_signal_length-AR_model_order,
    for j = 1:AR_model_order 
        z(i,j) = input_signal(i-j+AR_model_order);
    end 
    b(i) = input_signal(i+AR_model_order);
end
AR_parameters = b*pinv(z)';


output_signal_estimate = AR_parameters*z';  % estimated output
residual_error = b - output_signal_estimate; % inovative sequnce
residual_error_auto_covariance = xcorr(residual_error,'biased'); 
residual_error_auto_covariance = residual_error_auto_covariance(1:AR_model_order);    % auto-covariance

l = zeros(AR_model_order,2);
for i = 1:AR_model_order-1
    l(i,1) = -AR_parameters(i);
    l(i,2) = AR_parameters(1:AR_model_order-i) * AR_parameters(i+1:AR_model_order)';
end
l(AR_model_order,1) = -AR_parameters(AR_model_order); 
ar = l\residual_error_auto_covariance;

R_measurement_noise_covariance = sqrt(abs(ar(2))); 
Q_process_noise_covariance = ar(1)/R_measurement_noise_covariance - R_measurement_noise_covariance;
if (Q_process_noise_covariance<0) 
    Q_process_noise_covariance = 0; 
end

R_measurement_noise_std = sqrt(R_measurement_noise_covariance); 
Q_process_noise_std = sqrt(Q_process_noise_covariance);





function [x_aposteriori_state_estimate,C_upper_triangle_cholesky_factor_aposterioroi_covariance_matrix] =...
    kalman_filter_carlson_implementation(z_measurement,R_measurement_noise_variance,H_measurement_sensitivity_matrix...
    ,x_apriori_state_estimate,C_upper_triangle_cholesky_factor_of_apriori_covariance_matrix) 
% 
% Matlab implementation of Neil A. Carlson's ``square root'' 
% (Cholesky factor) implementation of the Kalman filter 
% 
% From the diskette included with 
% M. S. Grewal, L. R. Weill and A. P. Andrews 
% Global Positioning Systems, Inertial Navigation and Integration 
% John Wiley & Sons, 2000. 
% 
% INPUTS: 
%  z    measurement (SCALAR) 
%  R    variance of measurement error 
%  H    measurement sensitivity (row) vector 
%  xin  a priori estimate of state vector 
%  Cin  upper triangular Cholesky factor of covariance matrix 
%       of a priori state estimation uncertainty 
% OUTPUTS: 
%  xout a posteriori estimate of state vector 
%  Cout upper triangular Cholesky factor of covariance matrix 
%       of a posteriori state estimation uncertainty 
% 
C_upper_triangle_cholesky_factor_aposterioroi_covariance_matrix     = C_upper_triangle_cholesky_factor_of_apriori_covariance_matrix;  % Move for in-place (destructive) calculation. 
alpha = R_measurement_noise_variance; 
delta = z_measurement; 
  for j=1:length(x_apriori_state_estimate), 
  delta  = delta - H_measurement_sensitivity_matrix(j)*x_apriori_state_estimate(j); 
  sigma  = 0; 
    for i=1:j, 
    sigma  = sigma + C_upper_triangle_cholesky_factor_aposterioroi_covariance_matrix(i,j)*H_measurement_sensitivity_matrix(i); 
    end; 
  beta   = alpha; 
  alpha  = alpha + sigma^2; 
  gamma  = sqrt(alpha*beta); 
  eta    = beta/gamma; 
  zeta   = sigma/gamma; 
  w(j)   = 0; 
    for i=1:j; 
    tau    = C_upper_triangle_cholesky_factor_aposterioroi_covariance_matrix(i,j); 
    C_upper_triangle_cholesky_factor_aposterioroi_covariance_matrix(i,j) = eta*C_upper_triangle_cholesky_factor_aposterioroi_covariance_matrix(i,j) - zeta*w(i); 
    w(i)   = w(i) + tau*sigma; 
    end; 
  end; 
Cout    = C_upper_triangle_cholesky_factor_aposterioroi_covariance_matrix; 
epsilon = delta/alpha;     % Apply scaling to innovations, 
x_aposteriori_state_estimate    = x_apriori_state_estimate + epsilon*w'; % multiply by unscaled Kalman gain; 
return; 
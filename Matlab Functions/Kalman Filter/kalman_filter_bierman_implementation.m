function [x_aposteriori_state_estimate,U_upper_triangle_factor_aposteriori_covariance_matrix,D_diagonal_factor_aposteriori_covariance_matrix] = ...
    kalman_filter_bierman_implementation(z_measurement,...
                                        R_measurement_noise_variance,...
                                        H_measurement_sensitivity_matrix,...
                                        x_apriori_state_estimate,...
                                        U_upper_triangle_factor_covariance_matrix,...
                                        D_diagonal_factor_covariance_matrix) 
% 
% Matlab implementation of the 
% Bierman ``square root filtering without square roots'' 
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
%  Uin  unit upper triangular factor of covariance matrix of a priori state uncertainty 
%  Din  diagonal factor of covariance matrix of a priori state uncertainty 
% OUTPUTS: 
%  x    a posteriori estimate of state vector 
%  U    upper triangular UD factor of a posteriori state uncertainty covariance 
%  D    diagonal UD factor of a posteriori state uncertainty covariance 

x_aposteriori_state_estimate     = x_apriori_state_estimate;      % Store inputs into outputs,  
U_upper_triangle_factor_aposteriori_covariance_matrix     = U_upper_triangle_factor_covariance_matrix;      % because algorithm does in-place 
D_diagonal_factor_aposteriori_covariance_matrix     = D_diagonal_factor_covariance_matrix;      % (destructive) calculation of outputs. 
a     = U_upper_triangle_factor_aposteriori_covariance_matrix'*H_measurement_sensitivity_matrix';    % a is not modified, but 
b     = D_diagonal_factor_aposteriori_covariance_matrix*a;      % b is modified to become unscaled Kalman gain. 
dz    = z_measurement - H_measurement_sensitivity_matrix*x_apriori_state_estimate; 
alpha = R_measurement_noise_variance; 
gamma = 1/alpha; 
for j=1:length(x_apriori_state_estimate),
    beta   = alpha;
    alpha  = alpha + a(j)*b(j);
    lambda = -a(j)*gamma;
    gamma  = 1/alpha;
    D_diagonal_factor_aposteriori_covariance_matrix(j,j) = beta*gamma*D_diagonal_factor_aposteriori_covariance_matrix(j,j);
    for i=1:j-1,
        beta   = U_upper_triangle_factor_aposteriori_covariance_matrix(i,j);
        U_upper_triangle_factor_aposteriori_covariance_matrix(i,j) = beta + b(i)*lambda;
        b(i)   = b(i) + b(j)*beta;
    end;
end
dzs = gamma*dz;  % apply scaling to innovations 
x_aposteriori_state_estimate   = x_aposteriori_state_estimate + dzs*b; % multiply by unscaled Kalman gain 
return; 
 
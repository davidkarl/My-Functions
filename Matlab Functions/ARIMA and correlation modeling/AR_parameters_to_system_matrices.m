function [ A_state_transfer_matrix, H_measurement_matrix, Q_process_noise_covariance_matrix, R_measurement_covariance, P_state_estimate_covariance_matrix] = ...
    AR_parameters_to_system_matrices(...
                                        AR_parameters_signal,...
                                        g_signal,...
                                        auto_correlation_signal,...
                                        AR_parameters_noise,...
                                        g_noise,...
                                        auto_correlation_noise) 
% Given the AR generators for signal (as,bs), noise (an,bn), as well as the
% autocorrelations (cs,cn)
% this routine generates the appropriate state-space model to be used
% for Kalman filtering by kalmanrsi
% Usage: [A C Q r P]=ard2tme(as,bs,cs,an,bn,cn)
% A(q) y(t) = [B(q)/F(q)] u(t-nk) + [C(q)/D(q)] e(t)

%get AR model orders and total model order:
AR_model_order_signal = length(AR_parameters_signal) ;
AR_model_order_noise = length(AR_parameters_noise) ;
total_state_space_model_order = AR_model_order_signal+AR_model_order_noise ;

%Initialize state transfer matrix, process and estimation noise covariances, and output vector:
A_state_transfer_matrix = zeros(total_state_space_model_order) ; % system transfer matrix
Q_process_noise_covariance_matrix = zeros(total_state_space_model_order) ; % system noise covariance matrix
P_state_estimate_covariance_matrix = zeros(total_state_space_model_order) ; % initial state estimate covariance matrix
H_measurement_matrix = [zeros(1,AR_model_order_noise-1), 1 ,zeros(1,AR_model_order_signal-1), 1] ; % output matrix (vector, actually)

%Use AR parameters of signal and noise to get total state transfer matrix:
A_state_transfer_matrix(1:AR_model_order_noise,1:AR_model_order_noise) ...
    = AR_parameters_to_state_transfer_matrix(AR_parameters_noise) ;

A_state_transfer_matrix(AR_model_order_noise+1:total_state_space_model_order,AR_model_order_noise+1:total_state_space_model_order) ...
    = AR_parameters_to_state_transfer_matrix(AR_parameters_signal) ;


%Use AR models gains to get process noise covariance matrix:
Q_process_noise_covariance_matrix(AR_model_order_noise,AR_model_order_noise) = g_noise^2 ;
Q_process_noise_covariance_matrix(total_state_space_model_order,total_state_space_model_order) = g_signal^2 ;

%Use auto correlation sequences of noise and signal
P_state_estimate_covariance_matrix(1:AR_model_order_noise,1:AR_model_order_noise) = ...
    auto_correlation_to_toeplitz_matrix(auto_correlation_noise(1:AR_model_order_noise));

P_state_estimate_covariance_matrix(AR_model_order_noise+1:total_state_space_model_order,AR_model_order_noise+1:total_state_space_model_order) = ...
    auto_correlation_to_toeplitz_matrix(auto_correlation_signal(1:AR_model_order_signal)) ;

% r=trace(Q)*1e-6 ; % rather heuristic setting
R_measurement_covariance=1e-6 ;




